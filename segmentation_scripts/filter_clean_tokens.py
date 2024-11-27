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
from scipy.fft import fft
import scipy.stats as stats
from scipy.signal import find_peaks
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

def clean_digits(df: pd.DataFrame, filter_greater_than_numbers: bool, filter_implied_zeroes: bool, preidentified_periodical: bool) -> pd.DataFrame:
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
		console.print(f"Number of possible pages {len(possible_pages)} vs subset_digits {len(subset_digits)}")
		subset_digits = possible_pages
	non_digits_pages = df[(~df['token'].str.isdigit()) & (~df.page_number.isin(subset_digits.page_number))].copy()
	# just take the first page of non_digits_pages since we only need to keep one page per page_number
	non_digits_pages = non_digits_pages.groupby('page_number').first().reset_index()
	
	console.print(f"Number of digits in this volume: {len(subset_digits)}", style="bright_green")
	console.print(f"Number of non-digit pages in this volume: {len(non_digits_pages)}", style="bright_magenta")
	
	# Filter out tokens that are too large to be converted to integers
	subset_digits = subset_digits[subset_digits['token'].apply(lambda x: len(x) < 20)]  # Adjust the length as needed
	subset_digits['number'] = pd.to_numeric(subset_digits['token'], errors='coerce').fillna(0).astype(int)
	
	# Let's filter out the digits that are greater than the max possible page number or greater than the page number where they appear. Both of these are fairly subjective and can be adjusted as needed.
	if filter_greater_than_numbers:
		filtered_subset_digits = subset_digits[(subset_digits['number'] <= max_possible_page) & (subset_digits['number'] <= subset_digits.page_number)].copy()
		non_filtered_subset_digits = subset_digits[(subset_digits['number'] >= max_possible_page) & (~subset_digits.page_number.isin(filtered_subset_digits.page_number))].groupby('page_number').first().reset_index()
	else:
		filtered_subset_digits = subset_digits.copy()
		non_filtered_subset_digits = pd.DataFrame()

	console.print(f"Number of digits in this volume after filtering for max page length: {len(filtered_subset_digits)}", style="bright_green")
	console.print(f"Number of pages without digits in this volume after filtering for max page length: {len(non_filtered_subset_digits)}", style="bright_magenta")
	
	# Calculate implied zero only for digit pages
	filtered_subset_digits['implied_zero'] = filtered_subset_digits['page_number'].astype(int) - filtered_subset_digits['number']
	
	if filter_implied_zeroes:
		final_subset_digits = filtered_subset_digits[filtered_subset_digits['implied_zero'] >= 0]
		console.print(f"Number of digits in this volume after filtering for max page length and implied zero: {len(final_subset_digits)}", style="bright_green")
		remaining_missing_pages = df[(~df.page_number.isin(final_subset_digits.page_number)) & (~df.page_number.isin(non_digits_pages.page_number))].copy()
		remaining_missing_pages = remaining_missing_pages.groupby('page_number').first().reset_index()
		console.print(f"Number of pages without digits in this volume after filtering for max page length and digit pages: {len(remaining_missing_pages)}", style="bright_magenta")
	else:
		final_subset_digits = filtered_subset_digits.copy()
		remaining_missing_pages = pd.DataFrame()
	
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

	if preidentified_periodical:
		distribution_df = full_df_with_digits[['page_number', 'tokens_per_page', 'digits_per_page', 'start_issue']].drop_duplicates()

		distribution_df['digit_ratio'] = distribution_df['digits_per_page'] / distribution_df['tokens_per_page']

		# Calculate the mean digit ratio per issue
		mean_digit_ratio_per_issue = distribution_df.groupby('start_issue')['digit_ratio'].mean().reset_index(name='mean_digit_ratio')
		generate_table(mean_digit_ratio_per_issue, "Mean Digit Ratio per Issue")
	
	return full_df_with_digits

def process_file(file_path: str, is_preidentified_periodical: bool, should_filter_greater_than_numbers: bool, should_filter_implied_zeroes: bool) -> pd.DataFrame:
	full_df = read_csv_file(file_path)
	if 'page_number' not in full_df.columns:
		full_df = full_df.rename(columns={'page': 'page_number'})
	console.print(f"Volume has this many tokens: {len(full_df)}")
	if is_preidentified_periodical:
		console.print(f"Volume has this many issues: {full_df.start_issue.nunique()}")
		console.print(f"Volume has this many pages: {full_df.page_number.nunique()}")
	# Factorize the 'issue_number' column to create 'actual_issue_number'
	full_df = full_df.sort_values(by=['page_number'])

	if is_preidentified_periodical:
		full_df = full_df.rename(columns={'issue_number': 'original_issue_number', 'page_number': 'original_page_number'})
		full_df['temp_issue_number'] = pd.factorize(full_df['original_issue_number'])[0]

	else:
		full_df = full_df.rename(columns={'page_number': 'original_page_number'})
	# Factorize the 'original_page_number' column
	factorized_values, unique_values = pd.factorize(full_df['original_page_number'])

	# Adjust the factorized values to start at 1
	full_df['page_number'] = factorized_values + 1
	# Expand count column
	expanded_df = full_df.loc[full_df.index.repeat(full_df['count'])].reset_index(drop=True)
	console.print(f"Expanded volume has this many tokens: {len(expanded_df)}")

	# Calculate the number of tokens per page
	tokens_per_page = expanded_df.groupby('page_number').size().reset_index(name='tokens_per_page')

	# Merge the token counts back into the original DataFrame
	expanded_df = expanded_df.merge(tokens_per_page, on='page_number', how='left')
	missing_pages = full_df[~full_df.page_number.isin(expanded_df.page_number.unique())]
	expanded_df = pd.concat([expanded_df, missing_pages], ignore_index=True)
	expanded_df = expanded_df.reset_index(drop=True)

	if is_preidentified_periodical:
		annotated_df = full_df[['page_number', 'start_issue', 'end_issue', 'type_of_page']].drop_duplicates()

		# Group by 'start_issue' and aggregate
		grouped_df = annotated_df.groupby('start_issue').agg(
			first_page=('page_number', 'min'),
			last_page=('page_number', 'max'),
			number_of_pages=('page_number', 'count')
		).reset_index()
		grouped_df = grouped_df.sort_values(by='first_page')

		generate_table(grouped_df, "Grouped by 'start_issue'")
	else:
		grouped_df = pd.DataFrame()

	# Example usage
	subset_digits = clean_digits(expanded_df, should_filter_greater_than_numbers, should_filter_implied_zeroes, is_preidentified_periodical)
	subset_digits = subset_digits.sort_values(by=['page_number'])

	if is_preidentified_periodical:
		counts_per_annotated_issue = subset_digits.start_issue.value_counts().reset_index()

		generate_table(counts_per_annotated_issue, "Counts per Annotated Issue")
	return expanded_df, subset_digits, grouped_df

def get_volume_embeddings(file_path, preidentified_periodical, should_filter_greater_than_numbers, should_filter_implied_zeroes):
	expanded_df, subset_digits, grouped_df = process_file(file_path, preidentified_periodical, should_filter_greater_than_numbers, should_filter_implied_zeroes)
	if 'enumeration_chronology' not in expanded_df.columns:
		metadata_file_path = file_path.replace("_individual_tokens.csv", "_metadata.csv")
		metadata_df = read_csv_file(metadata_file_path)
		expanded_df = expanded_df.merge(metadata_df, on=['periodical_name', 'htid', 'record_url'], how='left')
	subset_expanded_df = expanded_df[['page_number', 'tokens_per_page', 'original_page_number', 'htid', 'title', 'pub_date', 'enumeration_chronology',
	   'type_of_resource', 'title', 'date_created', 'pub_date', 'language',
	   'access_profile', 'isbn', 'issn', 'lccn', 'oclc', 'page_count',
	   'feature_schema_version', 'access_rights', 'alternate_title',
	   'category', 'genre_ld', 'genre', 'contributor_ld', 'contributor',
	   'handle_url', 'source_institution_ld', 'source_institution', 'lcc',
	   'type', 'is_part_of', 'last_rights_update_date', 'pub_place_ld',
	   'pub_place', 'main_entity_of_page', 'publisher_ld', 'publisher', 'record_url', 'periodical_name',]].drop_duplicates()
	

	min_subset_digits = subset_digits[['original_page_number',
		'digits_per_page', 'page_number']].drop_duplicates()

	merged_expanded_df = subset_expanded_df.merge(min_subset_digits, on=['original_page_number', 'page_number'], how='left')
	merged_expanded_df['tokens_per_page'] = merged_expanded_df['tokens_per_page'].fillna(0)
	merged_expanded_df['digits_per_page'] = merged_expanded_df['digits_per_page'].fillna(0)
	merged_expanded_df['smoothed_tokens_per_page'] = merged_expanded_df['tokens_per_page'].rolling(window=3, center=True).mean()
	merged_expanded_df['smoothed_digits_per_page'] = merged_expanded_df['digits_per_page'].rolling(window=3, center=True).mean()
	# Scale the tokens per page and digits per page
	scaler = MinMaxScaler()

	merged_expanded_df['scaled_tokens_per_page'] = scaler.fit_transform(merged_expanded_df[['smoothed_tokens_per_page']])
	merged_expanded_df['scaled_digits_per_page'] = scaler.fit_transform(merged_expanded_df[['smoothed_digits_per_page']])

	merged_expanded_df['scaled_tokens_per_page'] = merged_expanded_df['scaled_tokens_per_page'].fillna(0)
	merged_expanded_df['scaled_digits_per_page'] = merged_expanded_df['scaled_digits_per_page'].fillna(0)


	console.print(merged_expanded_df[['page_number', 'tokens_per_page', 'smoothed_tokens_per_page', 'scaled_tokens_per_page', 'digits_per_page', 'smoothed_digits_per_page', 'scaled_digits_per_page']].head(2))

	# Perform FFT
	# Normalize signals for FFT and autocorrelation
	scaled_tokens = merged_expanded_df['scaled_tokens_per_page'].dropna().values
	scaled_digits = merged_expanded_df['scaled_digits_per_page'].dropna().values

	tokens_normalized = (scaled_tokens - np.mean(scaled_tokens)) / np.std(scaled_tokens)
	digits_normalized = (scaled_digits - np.mean(scaled_digits)) / np.std(scaled_digits)

	# Perform FFT
	tokens_fft = fft(tokens_normalized)
	digits_fft = fft(digits_normalized)
	frequencies = np.fft.fftfreq(len(tokens_fft))

	# Calculate dominant frequency
	dominant_frequency_index = np.argmax(np.abs(tokens_fft[1:len(frequencies)//2])) + 1
	dominant_frequency = frequencies[dominant_frequency_index]
	if dominant_frequency < 0:
		dominant_frequency = -dominant_frequency

	console.print(f"Dominant Frequency: {dominant_frequency}")

	# Detect Peaks in Signal
	# Calculate Autocorrelation
	tokens_autocorr = np.correlate(tokens_normalized, tokens_normalized, mode='full')
	digits_autocorr = np.correlate(digits_normalized, digits_normalized, mode='full')

	# Plot Autocorrelation
	lags = np.arange(-len(tokens_autocorr)//2, len(tokens_autocorr)//2)

	# Use find_peaks to locate all significant peaks
	tokens_peaks, _ = find_peaks(tokens_autocorr[len(tokens_autocorr)//2 + 1:], height=0.1)  # Skip lag 0
	digits_peaks, _ = find_peaks(digits_autocorr[len(digits_autocorr)//2 + 1:], height=0.1)  # Skip lag 0

	tokens_peak_lags = tokens_peaks + 1  # Adjust for indexing
	digits_peak_lags = digits_peaks + 1  # Adjust for indexing

	if len(tokens_peak_lags) > 0:
		tokens_autocorr_lag = tokens_peak_lags[0]  # Use first peak as dominant lag
	else:
		tokens_autocorr_lag = 1  # Default to 1 if no peaks are found

	if len(digits_peak_lags) > 0:
		digits_autocorr_lag = digits_peak_lags[0]  # Use first peak as dominant lag
	else:
		digits_autocorr_lag = 1  # Default to 1 if no peaks are found

	console.print(f"Likely issue length from tokens autocorrelation: {tokens_autocorr_lag}")
	console.print(f"Likely issue length from digits autocorrelation: {digits_autocorr_lag}")

	# Detect Likely Covers
	merged_expanded_df['is_likely_cover'] = (
		(merged_expanded_df['scaled_tokens_per_page'] < 0.2) &  # Low tokens
		(merged_expanded_df['scaled_digits_per_page'] < 0.2)    # Low digits
	)
	list_of_covers = merged_expanded_df[merged_expanded_df['is_likely_cover']].page_number.unique().tolist()
	# Aggregate Features for Embedding
	volume_features = {
		'avg_tokens': merged_expanded_df['tokens_per_page'].mean(),
		'avg_digits': merged_expanded_df['digits_per_page'].mean(),
		'dominant_frequency': dominant_frequency,
		'issue_length': tokens_autocorr_lag,  # Use the token-based autocorrelation lag
		'htid': merged_expanded_df['htid'].unique()[0],
		'periodical_name': merged_expanded_df['periodical_name'].unique()[0],
		'likely_covers': list_of_covers,
		'total_pages': merged_expanded_df['page_number'].nunique(),
	}
	volume_features_df = pd.DataFrame([volume_features])
	console.print(volume_features)
	return volume_features_df

def generate_token_frequency_analysis(should_filter_greater_than_numbers, should_filter_implied_zeroes):
	# Count the number of matching files
	matching_files = []
	for directory, _, files in tqdm(os.walk("../datasets/annotated_ht_ef_datasets/"), desc="Counting matching files"):
		for file in files:
			if file.endswith(".csv") and 'individual' in file:
				if os.path.exists(os.path.join(directory, file)):
					publication_name = directory.split("/")[-2]
					volume_number = directory.split("/")[-1]
					matching_files.append({"file": file, "directory": directory, "file_path": os.path.join(directory, file), "periodical_title": publication_name, "volume_directory": volume_number})
	matching_files_df = pd.DataFrame(matching_files)
	console.print(f"Found {len(matching_files_df)} matching files.", style="bright_green")

	data_directory_path = get_data_directory_path()
	preidentified_periodicals_df = read_csv_file(os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", "preidentified_periodicals_with_full_metadata.csv"))
	periodical_titles = preidentified_periodicals_df['lowercase_periodical_name'].unique()
	# volume_features_dfs = []
	for title in tqdm(periodical_titles, desc="Processing periodicals"):
		console.print(title)
		subset_preidentified_periodicals_df = preidentified_periodicals_df[preidentified_periodicals_df['lowercase_periodical_name'] == title]
		volumes = subset_preidentified_periodicals_df.volume_directory.unique()
		subset_matching_files_df = matching_files_df[matching_files_df['volume_directory'].isin(volumes)]
		is_annotated_periodical = True if len(subset_matching_files_df) > 0 else False
		for _, row in subset_preidentified_periodicals_df.iterrows():
			console.print(row.volume_directory)
			if pd.isna(row.volume_directory):
				continue
			matched_row = subset_matching_files_df[subset_matching_files_df['volume_directory'] == row.volume_directory]
			if len(matched_row) == 0:
				continue
			file_path = matched_row.file_path.values[0]  if is_annotated_periodical else os.path.join(data_directory_path,  "HathiTrust-pcc-datasets", row.publication_directory, row['volume_directory'], row['volume_directory'] + "_individual_tokens.csv")
			volume_features = get_volume_embeddings(file_path, is_annotated_periodical, should_filter_greater_than_numbers, should_filter_implied_zeroes)
			if os.path.exists("../datasets/volume_features_redo.csv"):
				volume_features.to_csv("../datasets/volume_features_redo.csv", mode='a', index=False, header=False)
			else:
				volume_features.to_csv("../datasets/volume_features_redo.csv", index=False)
	# volume_features_df = pd.concat(volume_features_dfs)
	# volume_features_df.to_csv("volume_features.csv", index=False)

if __name__ == "__main__":
	filter_greater_than_numbers = True
	filter_implied_zeroes = True
	generate_token_frequency_analysis(filter_greater_than_numbers, filter_implied_zeroes)