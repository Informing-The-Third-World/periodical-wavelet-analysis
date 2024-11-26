import os
import re
import pandas as pd
from tqdm import tqdm
import altair as alt
alt.data_transformers.disable_max_rows()
from collections import deque
import warnings
warnings.filterwarnings('ignore')
from rich.console import Console
from rich.table import Table
import numpy as np
import scipy.stats as stats
from minineedle import needle, core
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
console = Console()
import sys

sys.path.append("..")
from segmentation_scripts.utils import read_csv_file, get_data_directory_path

def generate_table(df: pd.DataFrame, table_title: str) -> None:
	"""
	Given a DataFrame, generate a Rich Table and print it to the console.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame to be printed.
	table_title : str
		The title of the table.
	"""
	# Create a Rich Table
	table = Table(title=table_title)
	columns = df.columns
	for column in columns:
		table.add_column(column.replace("_", " ").capitalize(), justify="center", style="cyan", no_wrap=True)    

	# Add rows to the table
	for _, row in df.iterrows():
		table.add_row(*[str(value) if pd.notna(value) else "" for value in row])

	# Print the table
	console.print(table)

def filter_integers(token: str) -> bool:
	"""Check if the token is an integer.
	
	Parameters
	----------
	token : str
		The token to be checked.

	Returns
	-------
	bool
		True if the token is an integer, False otherwise.
	"""
	return bool(re.match(r'^\d+$', token))

def calculate_digit_coverage(rows) -> int:
	"""Calculate the number of digits in the given rows.
	
	Parameters
	----------
	rows : pd.DataFrame
		The rows to be checked.
	
	Returns
	-------
	int
		The number of digits in the given rows.
		"""
	number_of_digits = rows['implied_zero'].notna().sum()
	return number_of_digits

def clean_digits(df: pd.DataFrame) -> pd.DataFrame:
	"""Clean and filter digit tokens in the DataFrame while retaining non-digit pages.
	
	Parameters
	----------
	df : pd.DataFrame
		The DataFrame to be cleaned.
		
	Returns
	-------
	pd.DataFrame
		The cleaned DataFrame.
	"""
	max_possible_page = df.page_number.max()
	
	df['token'] = df['token'].astype(str)
	if 'volume_name' in df.columns:
		df['volume_number'] = df['volume_number'].fillna(0)
	
	# Identify pages with digit tokens first using the 'isdigit' method
	subset_digits = df[df['token'].str.isdigit()].copy()
	# Also check using the filter_integers function which does regex matching
	possible_pages = subset_digits[subset_digits['token'].apply(filter_integers)].copy()
	# Use the smaller subset if possible_pages is smaller so that we are ensuring we are getting best quality digits
	if len(possible_pages) < len(subset_digits):
		subset_digits = possible_pages
	non_digits_pages = df[(~df['token'].str.isdigit()) & (~df.page_number.isin(subset_digits.page_number))].copy()
	# just take the first page of non_digits_pages since we only need to keep one page per page_number
	non_digits_pages = non_digits_pages.groupby('page_number').first().reset_index()
	
	
	
	console.print(f"Number of digits in this volume: {len(subset_digits)}", style="bright_green")
	console.print(f"Number of non-digit pages in this volume: {len(non_digits_pages)}", style="bright_magenta")
	
	subset_digits['number'] = subset_digits['token'].astype(int, errors='ignore')
	filtered_subset_digits = subset_digits[(subset_digits['number'] <= max_possible_page) & (subset_digits['number'] <= subset_digits.page_number)].copy()
	non_filtered_subset_digits = subset_digits[(subset_digits['number'] >= max_possible_page) & (~subset_digits.page_number.isin(filtered_subset_digits.page_number))].groupby('page_number').first().reset_index()
	console.print(f"Number of digits in this volume after filtering for max page length: {len(filtered_subset_digits)}", style="bright_green")
	console.print(f"Number of pages without digits in this volume after filtering for max page length: {len(non_filtered_subset_digits)}", style="bright_magenta")
	
	# Calculate implied zero only for digit pages
	filtered_subset_digits['implied_zero'] = filtered_subset_digits['page_number'].astype(int) - filtered_subset_digits['number']
	
	final_subset_digits = filtered_subset_digits[filtered_subset_digits['implied_zero'] >= 0]
	console.print(f"Number of digits in this volume after filtering for max page length and implied zero: {len(final_subset_digits)}", style="bright_green")
	remaining_missing_pages = df[(~df.page_number.isin(final_subset_digits.page_number)) & (~df.page_number.isin(non_digits_pages.page_number))].copy()
	remaining_missing_pages = remaining_missing_pages.groupby('page_number').first().reset_index()
	console.print(f"Number of pages without digits in this volume after filtering for max page length and digit pages: {len(remaining_missing_pages)}", style="bright_magenta")
	
	# Merge non-digit pages back into the DataFrame
	non_digits_pages['page_type'] = 'non_digit'
	remaining_missing_pages['page_type'] = 'negative_digit'
	final_subset_digits['page_type'] = 'digit'
	non_filtered_subset_digits['page_type'] = 'digit_too_large'
	full_df_with_digits = pd.concat([final_subset_digits, non_digits_pages, remaining_missing_pages, non_filtered_subset_digits]).sort_values(by=['page_number']).reset_index(drop=True)
	
	console.print(f"Number of pages after including non-digit pages: {full_df_with_digits.page_number.nunique()}", style="bright_yellow")

	if full_df_with_digits.page_number.nunique() != df.page_number.nunique():
		added_pages = df[~df.page_number.isin(full_df_with_digits.page_number)].copy()
		added_pages = added_pages.groupby('page_number').first().reset_index()
		added_pages['page_type'] = 'added'
		full_df_with_digits = pd.concat([full_df_with_digits, added_pages]).sort_values(by=['page_number']).reset_index(drop=True)
		console.print(f"Number of pages after including added pages: {full_df_with_digits.page_number.nunique()}", style="bright_yellow")

	# Calculate the number of digits per page
	tqdm.pandas(desc="Calculating digits per page")
	digits_per_page = full_df_with_digits.groupby('page_number').progress_apply(calculate_digit_coverage).reset_index(name='digits_per_page')
	full_df_with_digits = full_df_with_digits.merge(digits_per_page, on='page_number', how='left')

	distribution_df = full_df_with_digits[['page_number', 'tokens_per_page', 'digits_per_page', 'start_issue']].drop_duplicates()

	distribution_df['digit_ratio'] = distribution_df['digits_per_page'] / distribution_df['tokens_per_page']

	# Calculate the mean digit ratio per issue
	mean_digit_ratio_per_issue = distribution_df.groupby('start_issue')['digit_ratio'].mean().reset_index(name='mean_digit_ratio')
	generate_table(mean_digit_ratio_per_issue, "Mean Digit Ratio per Issue")
	
	return full_df_with_digits

def run_global_sequence_alignment(window: list, target_sequence: list, placeholder: int = -1) -> tuple:
	"""Apply global sequence alignment on the implied zero values within a window using minineedle, with placeholders.
	
	Parameters
	----------
	window : list
		The window containing the observed sequence.
		
	target_sequence : list
		The target sequence to be used in the alignment.
	
	placeholder : int
		The placeholder value to be used in the alignment.
		
	Returns
	-------
	tuple
		The alignment score, aligned observed sequence, and aligned target sequence.
	"""
	observed_sequence = [int(p[1]) if pd.notna(p[1]) else placeholder for p in window]
	
	# Check for valid entries in the observed sequence
	if all(val == placeholder for val in observed_sequence):
		return 0, [], []

	# Create Needleman-Wunsch global alignment instance
	alignment = needle.NeedlemanWunsch(observed_sequence, target_sequence)
	alignment.change_matrix(core.ScoreMatrix(match=6, miss=-0.5, gap=-1))

	try:
		# Run the alignment
		alignment.align()
		aligned_observed, aligned_target = alignment.get_aligned_sequences(core.AlignmentFormat.list)
		alignment_score = alignment.get_score()
		return alignment_score, aligned_observed, aligned_target

	except ZeroDivisionError:
		return 0, [], []

def select_likely_first_issue(df: pd.DataFrame, mean_threshold: float) -> pd.Series:
	"""Select the most likely first issue based on weighted scores.
	
	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the detected issue boundaries.
		
	mean_threshold : float
		The mean threshold size.
		
	Returns
	-------
	pd.Series
		The most likely first issue candidate.
	"""
	# Calculate frequency of start_page, end_page, and threshold_size
	start_page_freq = df['start_page'].value_counts(normalize=True).to_dict()
	end_page_freq = df['end_page'].value_counts(normalize=True).to_dict()
	# Define weights
	alpha, beta, gamma, delta = 0.4, 0.2, 0.2, 0.2

	# Add a column for weighted score
	def calculate_weighted_score(row):
		alignment_score = row['alignment_score']
		start_page_score = start_page_freq.get(row['start_page'], 0)
		end_page_score = end_page_freq.get(row['end_page'], 0)
		threshold_diff = abs(row['threshold_size'] - mean_threshold)
		threshold_score = 1 - (threshold_diff / mean_threshold)

		return (alpha * alignment_score +
				beta * start_page_score +
				gamma * end_page_score +
				delta * threshold_score)

	df['weighted_score'] = df.apply(calculate_weighted_score, axis=1)

	# Select the candidate with the highest weighted score
	best_candidate = df.sort_values(by='weighted_score', ascending=False).iloc[0]

	return best_candidate

def calculate_confidence_interval(df: pd.DataFrame, column: str, confidence: float = 0.95) -> tuple:
	"""Calculate the confidence interval for a given column.
	
	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the data.
		
	column : str
		The column for which the confidence interval is to be calculated.
		
	confidence : float
		The confidence level.
		
	Returns
	-------
	tuple
		The mean value, lower bound, upper bound, and margin of error.
	"""
	mean_val = df[column].mean()
	std_dev = df[column].std()
	n = len(df)

	# Calculate standard error
	standard_error = std_dev / np.sqrt(n)

	# Calculate confidence interval
	z_score = stats.norm.ppf((1 + confidence) / 2)
	margin_of_error = z_score * standard_error

	lower_bound = mean_val - margin_of_error
	upper_bound = mean_val + margin_of_error

	return mean_val, lower_bound, upper_bound, margin_of_error

def sequence_alignment_issue_detection_global(df: pd.DataFrame, threshold_sizes: list, placeholder: int = -1) -> pd.DataFrame:
	"""Detect issue boundaries using global sequence alignment.
	
	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the digit tokens.
		
	threshold_sizes : list
		The range of threshold sizes to be used for sequence alignment.
		
	placeholder : int
		The placeholder value to be used in the alignment.
	
	Returns
	-------
	pd.DataFrame
		The DataFrame containing the detected issue boundaries.
	"""
	df['page_number'] = df['page_number'].astype(int)
	df = df.sort_values(by=['page_number', 'implied_zero'])

	all_boundaries = []
	first_page_number = df.page_number.min()
	for threshold_size in tqdm(range(threshold_sizes[0], threshold_sizes[1]), desc="Running Sequence Alignment"):
		for additional_page in range(5):
			current_first_page_number = first_page_number + additional_page
			final_page_number = df[df.page_number == current_first_page_number + threshold_size]
			if final_page_number.empty:
				continue
			final_page_number = final_page_number.page_number.max()
			selected_rows = df[(df.page_number <= final_page_number) & (df.page_number >= current_first_page_number)]
			potential_sequence = list(zip(selected_rows['page_number'], selected_rows['implied_zero']))
			target_sequence = list(range(current_first_page_number, final_page_number))  # Generate the target sequence
			
			# Run sequence alignment with placeholders
			alignment_score, aligned_observed, aligned_target = run_global_sequence_alignment(potential_sequence, target_sequence, placeholder=placeholder)
			
			# Analyze the alignment score
			if alignment_score > 0:  # Adjust this threshold as needed
				all_boundaries.append((alignment_score, aligned_observed, aligned_target, threshold_size, current_first_page_number, final_page_number))

	boundaries_df = pd.DataFrame(all_boundaries, columns=['alignment_score', 'aligned_observed', 'aligned_target', 'threshold_size', 'start_page', 'end_page'])
	if boundaries_df.empty:
		console.print("No boundaries found.", style="bright_red")
		return boundaries_df, pd.DataFrame(), 0, 0
	seventy_five_threshold = boundaries_df['alignment_score'].quantile(0.75)
	top_boundaries = boundaries_df[boundaries_df.alignment_score > seventy_five_threshold].sort_values(by=['alignment_score', 'start_page'], ascending=[False, True])
	generate_table(top_boundaries[['alignment_score', 'threshold_size', 'start_page', 'end_page']], "Top Ten Likely First Issue Boundaries")

	# Calculate the mean of threshold sizes
	mean_threshold = top_boundaries['threshold_size'].mean()

	# Apply the selection function to the top ten boundaries
	best_first_issue = select_likely_first_issue(top_boundaries, mean_threshold)
	best_first_issue_df = pd.DataFrame([best_first_issue]).reset_index(drop=True)
	generate_table(best_first_issue_df[['alignment_score', 'threshold_size', 'start_page', 'end_page']], "Best First Issue Candidate")

	# Calculate confidence intervals for threshold_size and alignment_score
	mean_threshold, lower_threshold, upper_threshold, margin_error_threshold = calculate_confidence_interval(top_boundaries, 'threshold_size')
	mean_score, lower_score, upper_score, margin_error_score = calculate_confidence_interval(top_boundaries, 'alignment_score')

	console.print(f"Threshold Size: Mean = {mean_threshold}, CI = ({lower_threshold}, {upper_threshold}), Margin of Error = {margin_error_threshold}", style="bright_cyan")
	console.print(f"Alignment Score: Mean = {mean_score}, CI = ({lower_score}, {upper_score}), Margin of Error = {margin_error_score}", style="bright_cyan")
	return boundaries_df, top_boundaries, lower_threshold, upper_threshold 

def probabilistic_first_issue_detection(df: pd.DataFrame, threshold_sizes: list, window_size: int = 5, score_threshold: float = 0.5) -> pd.DataFrame:
	"""Identify the likely first issue length using probabilistic detection.
	
	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the digit tokens.
	
	threshold_sizes : list
		The range of threshold sizes to be used for sequence alignment.
		
	window_size : int
		The size of the sliding window.
	
	score_threshold : float
		The threshold score for issue detection.
		
	Returns
	-------
	pd.DataFrame
		The DataFrame containing the detected issue boundaries.
	"""
	df['page_number'] = df['page_number'].astype(int)
	df['implied_zero'] = df['implied_zero'].astype(int, errors='ignore')
	df = df.sort_values(by=['page_number'])

	all_boundaries = []
	first_page_number = df.page_number.min()
	for threshold_size in tqdm(range(threshold_sizes[0], threshold_sizes[1]), desc="Running Probabilistic Detection"):

		# Vary the start page within a defined range (similar to Needleman-Wunsch approach)
		for additional_page in range(5):
			current_first_page_number = first_page_number + additional_page
			final_page_number = df[df.page_number == current_first_page_number + threshold_size]
			if final_page_number.empty:
				continue
			final_page_number = final_page_number.page_number.max()
			selected_rows = df[(df.page_number <= final_page_number) & (df.page_number >= current_first_page_number)]

			sliding_window = deque(maxlen=window_size)
			cumulative_score = 0

			# Iterate through the observed sequence in the current window
			for _, row in selected_rows.iterrows():
				page_number = row['page_number']
				implied_zero = row['implied_zero'] if pd.notna(row['implied_zero']) else None
				section_weight = 0.2 if row['section'] != "body" else 0

				# Add to the sliding window
				if implied_zero is not None:
					sliding_window.append((page_number, implied_zero, section_weight))
				else:
					sliding_window.append((page_number, None, 0))

				# Calculate scores once the window is full
				non_none_values = [p for p in sliding_window if p[1] is not None]
				if len(sliding_window) == window_size and non_none_values:
					page_range = max(p[0] for p in sliding_window if p[1] is not None) - min(p[0] for p in sliding_window if p[1] is not None)
					implied_zero_diff = max(p[1] for p in sliding_window if p[1] is not None) - min(p[1] for p in sliding_window if p[1] is not None)

					score = 0
					if page_range > threshold_size:
						score += 0.7

					if implied_zero_diff > threshold_size:
						score += 0.5

					non_digit_count = sum(1 for p in sliding_window if p[1] is None)
					if non_digit_count > 0:
						score += 0.25 * (non_digit_count / window_size)

					section_weight = sum(p[2] for p in sliding_window)
					if section_weight > 0:
						score += 0.2 * section_weight

					# Accumulate scores and evaluate threshold
					cumulative_score += score
					if cumulative_score >= score_threshold:
						
						all_boundaries.append((
							cumulative_score, sliding_window, threshold_size,
							current_first_page_number, final_page_number
						))
						cumulative_score = 0  # Reset cumulative score

	boundaries_df = pd.DataFrame(all_boundaries, columns=[
		'cumulative_score', 'sliding_window', 'threshold_size', 'start_page', 'end_page'
	])
	# Analyze the top candidates for the first issue
	seventy_five_threshold = boundaries_df.describe()[['cumulative_score']].T['75%'].values[0]
	top_prob_candidates = boundaries_df[boundaries_df.cumulative_score > seventy_five_threshold].sort_values(by=['cumulative_score', 'start_page'], ascending=[False, True])
	generate_table(top_prob_candidates[['cumulative_score', 'threshold_size', 'start_page', 'end_page']], "Top Ten Probabilistic First Issue Candidates")
	return boundaries_df, top_prob_candidates

# Adjusted Raw Scores Initialization
def initialize_raw_scores(df, max_threshold):
	max_page = df['page_number'].max()
	if pd.isna(max_page):
		return np.zeros((0, 0), dtype=int)
	raw_scores = np.zeros((max_page + 1, max_threshold + 1), dtype=int)

	for _, row in df.iterrows():
		page = int(row['page_number'])
		number = int(row['implied_zero']) if row['page_type'] == 'digit' else 0
		
		if 0 <= page <= max_page and 0 <= number <= max_threshold:
			raw_scores[page, number] += 1

	return raw_scores

# Modified Prefix Sum Calculation for First Issue
def prefix_sums_first_issue(raw_scores, threshold_range, start_pages, updown=0.5, diag=0.25, otherwise=0.01, points=1.0):
	nrows, ncols = raw_scores.shape
	max_score_data = []

	# Iterate over threshold sizes
	for threshold_size in threshold_range:
		# Iterate over start pages extracted from the DataFrame
		for start_page in start_pages:
			end_page = start_page + threshold_size - 1

			# Ensure the end page doesn't exceed the matrix bounds
			if end_page >= nrows:
				continue
			
			# Initialize prefix sum matrix for the current configuration
			current_scores = raw_scores.copy()

			# Apply prefix sums within the current window
			for i in range(start_page, end_page + 1):
				for j in range(ncols):
					cell = otherwise + points * raw_scores[i, j]
					choices = []

					if j > 0:
						choices.append(current_scores[i, j-1] * updown)
					if i > start_page:
						choices.append(current_scores[i-1, j] * updown)
						if j > 0:
							choices.append(current_scores[i-1, j-1] * diag)

					cell += max(choices, default=0)
					current_scores[i, j] = cell

			# Calculate the total score for this configuration
			total_score = current_scores[start_page:end_page + 1, :].sum()

			# Collect the configuration and its total score
			max_score_data.append((total_score, threshold_size, start_page, end_page))

	return max_score_data

# Analyze Prefix Sum Results for First Issue
def detect_first_issue_prefix_sum(df, threshold_range=[10, 50], updown=0.5, diag=0.25, otherwise=0.01, points=1.0) -> tuple:
	"""
	Detect the first issue using prefix sums across different threshold sizes and start pages.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the digit tokens.

	threshold_range : list
		The range of threshold sizes to be used for sequence alignment.

	updown : float
		The weight for up/down movements.

	diag : float
		The weight for diagonal movements.

	otherwise : float
		The weight for other movements.

	points : float
		The weight for raw scores.

	Returns
	-------
	tuple
		The best candidate for the first issue and the DataFrame containing all candidates.
	"""
	max_score_data = []
	first_page_number = df.page_number.min()
	for threshold_size in tqdm(range(threshold_range[0], threshold_range[1]), desc="Running Prefix Sums"):
		# Vary the start page within a defined range (similar to Needleman-Wunsch approach)
		for additional_page in range(5):
			current_first_page_number = first_page_number + additional_page
			final_page_number = df[df.page_number == first_page_number + threshold_size]
			if final_page_number.empty:
				continue
			final_page_number = final_page_number.page_number.max()
			selected_rows = df[(df.page_number <= final_page_number) & (df.page_number >= current_first_page_number)]
			# Initialize raw scores matrix
			raw_scores = initialize_raw_scores(selected_rows, max_threshold=threshold_range[1])

			# Extract unique page numbers to use as start pages
			start_pages = selected_rows['page_number'].unique()

			# Run prefix sums across different thresholds and start pages
			max_score_data.extend(prefix_sums_first_issue(
				raw_scores,
				range(threshold_range[0], threshold_range[1]),
				start_pages,
				updown,
				diag,
				otherwise,
				points
			))

	# Convert to DataFrame
	results_df = pd.DataFrame(max_score_data, columns=['total_score', 'threshold_size', 'start_page', 'end_page'])
	results_df['threshold_size'] = results_df.end_page - results_df.start_page 
	best_candidate = results_df.sort_values(by='total_score', ascending=False).head(1)
	generate_table(best_candidate[['total_score', 'threshold_size', 'start_page', 'end_page']], "Best First Issue Candidate")

	# Analyze the top candidates for the first issue
	seventy_five_threshold = results_df.describe()[['total_score']].T['75%'].values[0]
	top_prob_candidates = results_df[results_df.total_score > seventy_five_threshold].sort_values(by=['total_score', 'start_page'], ascending=[False, True])
	generate_table(top_prob_candidates[['total_score', 'threshold_size', 'start_page', 'end_page']], "Top Prefix Sum First Issue Candidates")
	return best_candidate, results_df, top_prob_candidates

def calculate_combined_score(df, total_weight=0.4, alignment_weight=0.3, cumulative_weight=0.3) -> pd.DataFrame:
	"""Calculate a combined score based on total_score, alignment_score, and cumulative_score.
	
	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing the scores.
		
	total_weight : float
		The weight for the total score.
		
	alignment_weight : float
		The weight for the alignment score.
		
	cumulative_weight : float
		The weight for the cumulative score.
		
	Returns
	-------
	pd.DataFrame
		The DataFrame containing the combined score.
	"""
	# Normalize the scores using Min-Max Scaling
	scaler = MinMaxScaler()

	df[['norm_total_score', 'norm_alignment_score', 'norm_cumulative_score']] = scaler.fit_transform(
		df[['total_score', 'alignment_score', 'cumulative_score']]
	)

	# Calculate the combined score as a weighted sum
	df['combined_score'] = (
		df['norm_total_score'] * total_weight +
		df['norm_alignment_score'] * alignment_weight +
		df['norm_cumulative_score'] * cumulative_weight
	)

	return df

def generate_issue_binary(start_page, end_page, total_pages) -> np.ndarray:
	"""
	Convert the issue boundaries into a binary format indicating issue presence.

	Parameters
	----------
	start_page : int
		The starting page of the issue.

	end_page : int
		The ending page of the issue.

	total_pages : int
		The total number of pages in the volume.

	Returns
	-------
	np.ndarray
		The binary representation of the issue boundaries.
	"""
	issue_binary = np.zeros(int(total_pages), dtype=int)
	issue_binary[int(start_page):int(end_page) + 1] = 1
	return issue_binary

def calculate_first_issue_accuracy(top_issues_df, grouped_df, total_pages) -> pd.DataFrame:
	"""
	Calculate accuracy, precision, recall, and F1-score for first issue detection.

	Parameters
	----------
	top_issues_df : pd.DataFrame
		The DataFrame containing the top issue candidates.

	grouped_df : pd.DataFrame
		The DataFrame containing the grouped issue information.

	total_pages : int
		The total number of pages in the volume.

	Returns
	-------
	pd.DataFrame
		The DataFrame containing the calculated metrics.
	"""
	# Extract the first row from top_issues_df as the predicted first issue
	metrics_df = top_issues_df.copy()
	for index, row in tqdm(top_issues_df.iterrows(), total=top_issues_df.shape[0], desc="Calculating Metrics"):
		# Extract the predicted first issue boundaries
		predicted_start_page = int(row['start_page'])
		predicted_end_page = int(row['end_page'])
		# Extract the actual first issue boundaries from grouped_df
		actual_first_issue = grouped_df.iloc[0]
		actual_start_page = int(actual_first_issue['first_page'])
		actual_end_page = int(actual_first_issue['last_page'])
		actual_issue_length = int(actual_first_issue['number_of_pages'])

		# Convert predicted and actual issues to binary format
		predicted_issues_binary = generate_issue_binary(predicted_start_page, predicted_end_page, total_pages)
		actual_issues_binary = generate_issue_binary(actual_start_page, actual_end_page, total_pages)

		# Calculate accuracy, precision, recall, and F1-score
		accuracy = accuracy_score(actual_issues_binary, predicted_issues_binary)
		precision = precision_score(actual_issues_binary, predicted_issues_binary)
		recall = recall_score(actual_issues_binary, predicted_issues_binary)
		f1 = f1_score(actual_issues_binary, predicted_issues_binary)

		metrics_df.loc[index, 'accuracy'] = accuracy
		metrics_df.loc[index, 'precision'] = precision
		metrics_df.loc[index, 'recall'] = recall
		metrics_df.loc[index, 'f1'] = f1
		metrics_df.loc[index, 'actual_start_page'] = actual_start_page
		metrics_df.loc[index, 'actual_end_page'] = actual_end_page
		metrics_df.loc[index, 'actual_total_volume_pages'] = total_pages
		metrics_df.loc[index, 'actual_issue_length'] = actual_issue_length
	return metrics_df

def load_and_expand_data(file_path: str) -> tuple:
	"""Load the data from the given file and expand the count column.

	Parameters
	----------
	file_path : str
		The path to the file to be loaded.

	Returns
	-------
	tuple
		The original DataFrame and the expanded DataFrame.
	"""
	full_df = read_csv_file(file_path)
	full_df = full_df.sort_values(by=['page_number'])
	full_df = full_df.rename(columns={'issue_number': 'original_issue_number'})
	full_df['temp_issue_number'] = pd.factorize(full_df['original_issue_number'])[0]
	console.print(f"Volume has this many tokens: {len(full_df)}", style="bright_blue")
	console.print(f"Volume has this many issues: {full_df.start_issue.nunique()}", style="bright_blue")
	console.print(f"Volume has this many pages: {full_df.page_number.nunique()}", style="bright_blue")
	expanded_df = full_df.loc[full_df.index.repeat(full_df['count'])].reset_index(drop=True)
	console.print(f"Expanded volume has this many tokens: {len(expanded_df)}", style="bright_blue")
	tokens_per_page = expanded_df.groupby('page_number').size().reset_index(name='tokens_per_page')
	expanded_df = expanded_df.merge(tokens_per_page, on='page_number', how='left')

	missing_pages = full_df[~full_df.page_number.isin(expanded_df.page_number.unique())]
	if len(missing_pages) > 0:
		expanded_df = pd.concat([expanded_df, missing_pages], ignore_index=True)
		expanded_df = expanded_df.reset_index(drop=True)

	return full_df, expanded_df

def process_annotated_volumes(rerun_code: bool) -> None:
	"""Process the annotated volumes to identify the first issue.
	
	Parameters
	----------
	rerun_code : bool
		Flag to indicate whether to rerun the code.
	
	Returns
	-------
	None
	"""
	# Count the number of matching files
	matching_files = []
	for directory, _, files in tqdm(os.walk("../datasets/annotated_ht_ef_datasets/"), desc="Counting matching files"):
		for file in files:
			if file.endswith(".csv") and 'individual' in file:
				if os.path.exists(os.path.join(directory, file)):
					matching_files.append({"file": file, "directory": directory, "file_path": os.path.join(directory, file)})
	matching_files_df = pd.DataFrame(matching_files)
	console.print(f"Found {len(matching_files_df)} matching files.", style="bright_green")

	for index, row in matching_files_df.iterrows():
		file = row['file']
		directory = row['directory']
		file_path = row['file_path']
		console.print(f"Processing file: {file_path}. Number {index} out of {len(matching_files_df)}", style="bright_white")

		first_issue_directory = directory.replace("annotated_ht_ef_datasets", "first_issue_metrics")
		if os.path.exists(first_issue_directory) == False:
			os.makedirs(first_issue_directory, exist_ok=True)
		metrics_file_output_path = os.path.join(first_issue_directory, file.replace(".csv", "_first_issue_metrics.csv"))
		if os.path.exists(metrics_file_output_path) and not rerun_code:
			console.print(f"Metrics file already exists for {file_path}. Skipping.", style="bright_red")
			continue
		# Load and expand data
		full_df, expanded_df = load_and_expand_data(file_path)
		annotated_df = full_df[['page_number', 'start_issue', 'end_issue', 'type_of_page']].drop_duplicates()
		# Group by 'start_issue' and aggregate
		grouped_df = annotated_df.groupby('start_issue').agg(
			first_page=('page_number', 'min'),
			last_page=('page_number', 'max'),
			number_of_pages=('page_number', 'count')
		).reset_index()
		grouped_df = grouped_df.sort_values(by='first_page')
		if len(grouped_df) <= 1:
			console.print("Only one issue found. Skipping volume.", style="bright_red")
			continue
		annotated_first_issue = grouped_df[0:1]
		if annotated_first_issue.number_of_pages.values[0] <= 10:
			console.print("First issue has less than 10 pages. Skipping volume.", style="bright_red")
			continue

		subset_digits = clean_digits(expanded_df)
		counts_per_annotated_issue = subset_digits.start_issue.value_counts().reset_index()
		generate_table(counts_per_annotated_issue, "Counts per Annotated Issue")

		dedup_subset_digits = subset_digits.drop_duplicates()
		sequence_alignment_full_data = False
		sequence_alignment_likely_first_issue_boundaries_df, top_sequence_alignment_boundaries_df, lower_threshold, upper_threshold = sequence_alignment_issue_detection_global(dedup_subset_digits, threshold_sizes=[10, 200], placeholder=-1)
		console.print(f"Top sequence alignment boundaries: {len(top_sequence_alignment_boundaries_df)}", style="bright_white")
		if (len(top_sequence_alignment_boundaries_df) > 150) or (len(top_sequence_alignment_boundaries_df) == 0):
			console.print("Too many candidates found. Running with full data.", style="bright_red")
			sequence_alignment_full_data = True
			sequence_alignment_likely_first_issue_boundaries_df, top_sequence_alignment_boundaries_df, lower_threshold, upper_threshold = sequence_alignment_issue_detection_global(subset_digits, threshold_sizes=[10, 200], placeholder=-1)

		sliding_window_prob_first_issue_df, top_sliding_window_prob_candidates_df = probabilistic_first_issue_detection(subset_digits, threshold_sizes=[round(lower_threshold), round(upper_threshold)], window_size=5, score_threshold=0.5)

		best_first_issue, prefix_all_candidates_df, top_prefix_prob_candidates_df = detect_first_issue_prefix_sum(subset_digits, threshold_range=[round(lower_threshold), round(upper_threshold)])

		top_issues = prefix_all_candidates_df[['total_score', 'threshold_size', 'start_page', 'end_page']].merge(sequence_alignment_likely_first_issue_boundaries_df[['alignment_score', 
		'threshold_size', 'start_page', 'end_page']], on=['threshold_size', 'start_page', 'end_page'], how='inner').sort_values(by=['total_score', 'alignment_score'], ascending=[False, False])

		if top_issues.empty:
			top_issues = prefix_all_candidates_df[['total_score', 'threshold_size', 'start_page', 'end_page']].merge(sequence_alignment_likely_first_issue_boundaries_df[['alignment_score', 'threshold_size', 'start_page', 'end_page']], on=['threshold_size', 'start_page', 'end_page'], how='outer').sort_values(by=['total_score', 'alignment_score'], ascending=[False, False])

		top_issues_df = top_issues.merge(sliding_window_prob_first_issue_df[['cumulative_score','threshold_size', 'start_page', 'end_page']], on=['threshold_size', 'start_page', 'end_page'], how='inner').sort_values(by=['total_score', 'cumulative_score'], ascending=[False, False])

		if top_issues_df.empty:
			top_issues_df = top_issues.merge(sliding_window_prob_first_issue_df[['cumulative_score','threshold_size', 'start_page', 'end_page']], on=['threshold_size', 'start_page', 'end_page'], how='outer').sort_values(by=['total_score', 'cumulative_score'], ascending=[False, False])

		if top_issues_df.empty:
			console.print("No common candidates found between the three methods.", style="bright_red")
			continue

		top_issues_df = top_issues_df.drop_duplicates()
		generate_table(top_issues_df[['threshold_size', 'start_page', 'end_page', 'alignment_score', 'total_score', 'cumulative_score']], "Top First Issue Candidates")

		weights = {
			'total_score': 0.4,
			'alignment_score': 0.4,
			'cumulative_score': 0.2
		}

		top_issues_df['composite_score'] = (
			weights['total_score'] * top_issues_df['total_score'] +
			weights['alignment_score'] * top_issues_df['alignment_score'] +
			weights['cumulative_score'] * top_issues_df['cumulative_score']
		)

		top_issues_df = top_issues_df.sort_values(by='composite_score', ascending=False)
		generate_table(top_issues_df[['threshold_size', 'start_page', 'end_page', 'alignment_score', 'total_score', 'cumulative_score', 'composite_score']], "Top First Issue Candidates with Composite Score")

		# Apply the function to your DataFrame
		top_issues_df = calculate_combined_score(top_issues_df)

		# Define the total number of pages in the volume
		total_pages = full_df['page_number'].max() + 1

		# Calculate accuracy, precision, recall, and F1-score for the first issue detection
		metrics_df = calculate_first_issue_accuracy(top_issues_df, grouped_df, total_pages)

		metrics_df = metrics_df.sort_values(by='f1', ascending=False)
		generate_table(metrics_df[['threshold_size', 'start_page', 'end_page', 'accuracy', 'precision', 'recall', 'f1']], "First Issue Detection Metrics")
		metrics_df['annotated_file_path'] = os.path.join(directory, file)
		metrics_df['sequence_alignment_full_data'] = sequence_alignment_full_data
		metrics_df['final_number_of_candidates'] = len(top_issues_df)
		metrics_df['upper_threshold'] = upper_threshold
		metrics_df['lower_threshold'] = lower_threshold
		metrics_df['sequence_alignment_candidates'] = len(top_sequence_alignment_boundaries_df)
		metrics_df['probabilistic_candidates'] = len(top_sliding_window_prob_candidates_df)
		metrics_df['prefix_sum_candidates'] = len(top_prefix_prob_candidates_df)
		
		metrics_df.to_csv(metrics_file_output_path, index=False)


if __name__ == "__main__":
	reprocess_data = True
	process_annotated_volumes(reprocess_data)