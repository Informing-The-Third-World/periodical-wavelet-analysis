import os
import re
import pandas as pd
from tqdm import tqdm
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
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
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
console = Console()
import sys

sys.path.append("..")
from segmentation_scripts.utils import read_csv_file, get_data_directory_path, save_chart

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
		if 'start_issue' not in full_df.columns:
			console.print(f"Periodical identified as annotated but missing start_issue, with htid {full_df.htid.unique()}", style="bright_magenta")
			should_continue = console.input("Do you want to continue the code? y/n")
			if should_continue == 'n':
				console.print(file_path)
				console.print(full_df.columns)
				sys.exit()
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

def calculate_dominant_frequency(tokens_normalized: np.ndarray, tokens_mean: float, tokens_std: float, page_type: str, min_tokens: float) -> tuple:
    """
    Calculate the dominant frequency of the given signal and a dynamic cutoff.

    Parameters
    ----------
    tokens_normalized : np.ndarray
        The normalized signal to be analyzed.
    tokens_mean : float
        Mean of the original tokens per page.
    tokens_std : float
        Standard deviation of the original tokens per page.
    page_type : str
        The type of page being analyzed (e.g., "tokens_per_page").
    min_tokens : float
        The minimum observed tokens per page in the original scale.

    Returns
    -------
    tuple
        - dominant_frequency: The dominant frequency of the signal.
        - positive_frequencies: Array of positive frequencies.
        - positive_amplitudes: Array of positive amplitudes.
        - dynamic_cutoff_original_scale: Dynamic cutoff in the original scale.
    """
    # Perform FFT
    tokens_fft = fft(tokens_normalized)
    frequencies = np.fft.fftfreq(len(tokens_fft))

    # Only look at positive frequencies
    positive_frequencies = frequencies[:len(frequencies)//2]
    positive_amplitudes = np.abs(tokens_fft[:len(tokens_fft)//2])

    # Find the dominant frequency (ignoring the DC component)
    peaks, _ = find_peaks(positive_amplitudes[1:])
    if len(peaks) > 0:
        dominant_frequency_index = peaks[np.argmax(positive_amplitudes[1:][peaks])] + 1
        dominant_frequency = positive_frequencies[dominant_frequency_index]
    else:
        dominant_frequency = None  # Handle cases with no significant peaks

    # Calculate a dynamic cutoff
    if len(peaks) > 0:
        peak_amplitude = np.max(positive_amplitudes[1:][peaks])
        dynamic_cutoff_normalized = np.median(tokens_normalized) - peak_amplitude
        dynamic_cutoff_normalized = max(dynamic_cutoff_normalized, np.percentile(tokens_normalized, 10))  # Ensure it's not too low
    else:
        dynamic_cutoff_normalized = np.percentile(tokens_normalized, 10)  # Fallback to 10th percentile if no peaks

    # Convert the dynamic cutoff back to the original scale
    dynamic_cutoff_original_scale = (dynamic_cutoff_normalized * tokens_std) + tokens_mean

    # Clamp the dynamic cutoff to be at least the minimum tokens per page
    dynamic_cutoff_original_scale = max(dynamic_cutoff_original_scale, min_tokens)

    console.print(f"Dominant Frequency: {dominant_frequency} for {page_type}")
    console.print(f"Dynamic Cutoff (Original Scale): {dynamic_cutoff_original_scale} for {page_type}")

    return dominant_frequency, positive_frequencies, positive_amplitudes, dynamic_cutoff_original_scale

def process_tokens(file_path: str, preidentified_periodical: bool, should_filter_greater_than_numbers: bool, should_filter_implied_zeroes: bool) -> tuple:
	"""
	Process tokens from the given file and return the processed DataFrame along with normalized token and digit signals.

	Parameters
	----------
	file_path : str
		The path to the file containing the token data.
	preidentified_periodical : bool
		Flag indicating whether the periodical is pre-identified.
	should_filter_greater_than_numbers : bool
		Flag indicating whether to filter out numbers greater than the max possible page number.
	should_filter_implied_zeroes : bool
		Flag indicating whether to filter out implied zeroes.

	Returns
	-------
	tuple
		A tuple containing the processed DataFrame, normalized token signals, and normalized digit signals.
	"""
	expanded_df, subset_digits, grouped_df = process_file(file_path, preidentified_periodical, should_filter_greater_than_numbers, should_filter_implied_zeroes)
	
	# Merge metadata if not already present
	if 'enumeration_chronology' not in expanded_df.columns:
		metadata_file_path = file_path.replace("_individual_tokens.csv", "_metadata.csv")
		metadata_df = read_csv_file(metadata_file_path)
		expanded_df = expanded_df.merge(metadata_df, on=['periodical_name', 'htid', 'record_url'], how='left')
	
	subset_cols = ['page_number', 'tokens_per_page', 'original_page_number', 'htid', 'title', 'pub_date', 'enumeration_chronology', 'type_of_resource', 'title', 'date_created', 'pub_date', 'language', 'access_profile', 'isbn', 'issn', 'lccn', 'oclc', 'page_count', 'feature_schema_version', 'access_rights', 'alternate_title', 'category', 'genre_ld', 'genre', 'contributor_ld', 'contributor', 'handle_url', 'source_institution_ld', 'source_institution', 'lcc', 'type', 'is_part_of', 'last_rights_update_date', 'pub_place_ld', 'pub_place', 'main_entity_of_page', 'publisher_ld','publisher', 'record_url', 'periodical_name'] 
	if preidentified_periodical:
		subset_cols = subset_cols + ['start_issue', 'end_issue', 'type_of_page']
	# Select relevant columns and drop duplicates
	subset_expanded_df = expanded_df[subset_cols].drop_duplicates()
	
	min_subset_digits = subset_digits[['original_page_number', 'digits_per_page', 'page_number']].drop_duplicates()
	
	# Merge the token and digit data
	merged_expanded_df = subset_expanded_df.merge(min_subset_digits, on=['original_page_number', 'page_number'], how='left')
	merged_expanded_df['tokens_per_page'] = merged_expanded_df['tokens_per_page'].fillna(0)
	merged_expanded_df['digits_per_page'] = merged_expanded_df['digits_per_page'].fillna(0)
	
	# Smooth the data
	merged_expanded_df['smoothed_tokens_per_page'] = merged_expanded_df['tokens_per_page'].rolling(window=3, center=True).mean()
	merged_expanded_df['smoothed_digits_per_page'] = merged_expanded_df['digits_per_page'].rolling(window=3, center=True).mean()
	
	# Scale the data
	scaler = MinMaxScaler()
	merged_expanded_df['scaled_tokens_per_page'] = scaler.fit_transform(merged_expanded_df[['smoothed_tokens_per_page']])
	merged_expanded_df['scaled_digits_per_page'] = scaler.fit_transform(merged_expanded_df[['smoothed_digits_per_page']])
	
	merged_expanded_df['scaled_tokens_per_page'] = merged_expanded_df['scaled_tokens_per_page'].fillna(0)
	merged_expanded_df['scaled_digits_per_page'] = merged_expanded_df['scaled_digits_per_page'].fillna(0)
	
	generate_table(merged_expanded_df[['page_number', 'tokens_per_page', 'smoothed_tokens_per_page', 'scaled_tokens_per_page', 'digits_per_page', 'smoothed_digits_per_page', 'scaled_digits_per_page']].head(2), "Token and Digit Data")
	
	# Normalize signals for FFT and autocorrelation
	scaled_tokens = merged_expanded_df['scaled_tokens_per_page'].dropna().values
	scaled_digits = merged_expanded_df['scaled_digits_per_page'].dropna().values
	
	tokens_normalized = (scaled_tokens - np.mean(scaled_tokens)) / np.std(scaled_tokens)
	digits_normalized = (scaled_digits - np.mean(scaled_digits)) / np.std(scaled_digits)

	tokens_mean = merged_expanded_df['tokens_per_page'].mean()
	digits_mean = merged_expanded_df['digits_per_page'].mean()
	tokens_std = merged_expanded_df['tokens_per_page'].std()
	digits_std = merged_expanded_df['digits_per_page'].std()
	
	return merged_expanded_df, grouped_df, tokens_normalized, digits_normalized, tokens_mean, tokens_std, digits_mean, digits_std

def plot_volume_frequencies_matplotlib(volume_frequencies: list, periodical_name: str, output_dir: str):
	"""
	Plot all volume frequencies on the same graph and save as an image.

	Parameters:
	- volume_frequencies: List of volume frequency data.
	- periodical_name: Name of the periodical for the title.
	- output_dir: Directory to save the plot.
	"""
	plt.figure(figsize=(14, 8))
	for volume in volume_frequencies:
		plt.plot(
			volume['tokens_positive_frequencies'], 
			volume['tokens_positive_amplitudes'], 
			label=f"Volume {volume['htid']}"
		)
	plt.title(f'Frequency Spectra of All Volumes in {periodical_name}')
	plt.xlabel('Frequency')
	plt.ylabel('Amplitude')
	plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside plot
	plt.tight_layout()
	plt.savefig(f"{output_dir}/amplitude_vs_frequencies/{periodical_name}_volume_frequencies.png", dpi=300)  # Save at high resolution
	plt.close()  # Close the plot to save memory

def plot_tokens_per_page(volume_frequencies: list, output_dir: str, periodical_name: str):
	"""
	Plot tokens per page over pages for all volumes.

	Parameters:
	- volume_frequencies: List of volume frequency data.
	- output_dir: Directory to save the plot.
	- periodical_name: Name of the periodical for the title.
	"""
	plt.figure(figsize=(14, 8))
	
	for volume in volume_frequencies:
		# Extract tokens per page and page numbers
		tokens_per_page = volume['tokens_per_page']
		page_numbers = volume['page_numbers']

		plt.plot(page_numbers, tokens_per_page, label=f"Volume {volume['htid']}")
	
	plt.title(f'Tokens Per Page Across Volumes in {periodical_name}')
	plt.xlabel('Page Number')
	plt.ylabel('Tokens Per Page')
	plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside plot
	plt.tight_layout()
	plt.savefig(f"{output_dir}/tokens_per_page/{periodical_name}_tokens_per_page.png", dpi=300)  # Save at high resolution
	plt.close()

def check_if_actual_issue(row, grouped_df):
    """
    Check if the given row corresponds to an actual issue.

    Parameters
    ----------
    row : pd.Series
        The row of the DataFrame being checked.
    grouped_df : pd.DataFrame
        Grouped DataFrame with issue boundaries.

    Returns
    -------
    bool
        True if the page is part of an actual issue, False otherwise.
    """
    subset_grouped_df = grouped_df[grouped_df.first_page == row.page_number]
    return len(subset_grouped_df) > 0

def visualize_annotated_periodicals(merged_expanded_df, grouped_df, output_dir, periodical_name, dynamic_cutoff):
	"""
	Visualize tokens per page for annotated periodicals and calculate the lowest threshold.

	Parameters:
	- expanded_df: The expanded DataFrame with tokens per page.
	- grouped_df: Grouped DataFrame with issue boundaries.
	- output_dir: Directory to save the visualization.
	- periodical_name: Name of the periodical.
	"""


	# Create the base Altair chart
	selection = alt.selection_point(fields=['start_issue'], bind='legend')
	base = alt.Chart(merged_expanded_df[['page_number', 'tokens_per_page', 'start_issue', 'htid']]).mark_line(point=True).encode(
		x=alt.X("page_number:Q", scale=alt.Scale(zero=False)),
		y=alt.Y('tokens_per_page:Q', scale=alt.Scale(zero=False)),
		color='start_issue:N',
		opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
		tooltip=['page_number', 'tokens_per_page', 'start_issue', 'htid']
	).add_params(selection).properties(
		width=600,
		height=300,
		title=f'Tokens per Page per Issue - {periodical_name} for Volume {merged_expanded_df.htid.unique()[0]}'
	)

	# Add the dynamic cutoff line
	cutoff_line = alt.Chart(pd.DataFrame({'y': [dynamic_cutoff]})).mark_rule(color='blue').encode(
		y=alt.Y('y:Q', axis=alt.Axis(title=None))
	)

	# Combine the base chart and the cutoff line
	chart = base + cutoff_line
	# Identify pages below the dynamic cutoff
	lowest_tokens_df = merged_expanded_df[merged_expanded_df['tokens_per_page'] <= dynamic_cutoff]

	tqdm.pandas(desc="Checking if actual issue")
	# Add 'actual_issue' column to the DataFrame
	lowest_tokens_df['actual_issue'] = lowest_tokens_df.progress_apply(
		check_if_actual_issue, args=(grouped_df,), axis=1
	)
	generate_table(lowest_tokens_df[lowest_tokens_df.actual_issue == True], "Lowest Tokens per Page")

	# Sort and print missing issues
	missing_issues = grouped_df[
		(~grouped_df.start_issue.isin(
			lowest_tokens_df[
				lowest_tokens_df.actual_issue == True
			].start_issue
		))
	]
	print(f"We are missing the following issues: {missing_issues.start_issue.unique()}")
	return missing_issues.start_issue.unique().tolist(), chart

def generate_volume_embeddings(volume_paths_df: pd.DataFrame, output_dir: str, run_correlations: bool) -> pd.DataFrame:
	volume_frequencies = []
	volume_paths_df = volume_paths_df.reset_index(drop=True)
	volume_paths_df = volume_paths_df.sort_values(by=['table_row_index'])
	periodical_name = volume_paths_df['lowercase_periodical_name'].unique()[0]
	altair_charts = []
	for _, volume in volume_paths_df.iterrows():
		merged_expanded_df, grouped_df, tokens_normalized, digits_normalized, tokens_mean, tokens_std, digits_mean, digits_std = process_tokens(
			volume['file_path'], 
			volume['is_annotated_periodical'], 
			volume['should_filter_greater_than_numbers'], 
			volume['should_filter_implied_zeroes']
		)
		
		# Calculate dominant frequencies for tokens and digits
		tokens_dominant_frequency, tokens_positive_frequencies, tokens_positive_amplitudes, tokens_dynamic_cutoff = calculate_dominant_frequency(
			tokens_normalized, tokens_mean, tokens_std, "tokens_per_page",  merged_expanded_df['tokens_per_page'].min()
		)
		digits_dominant_frequency, digits_positive_frequencies, digits_positive_amplitudes, digits_dynamic_cutoff = calculate_dominant_frequency(digits_normalized, digits_mean, digits_std, "digits_per_page", merged_expanded_df['digits_per_page'].min())
		
		if volume['is_annotated_periodical'] and len(grouped_df) > 1:
			missing_issues, chart = visualize_annotated_periodicals(merged_expanded_df, grouped_df, output_dir, volume['lowercase_periodical_name'], tokens_dynamic_cutoff)
			altair_charts.append(chart)
		else:
			missing_issues = []
			chart = None

		# Use dynamic cutoffs for tokens and digits
		merged_expanded_df['is_likely_cover'] = (
			(merged_expanded_df['tokens_per_page'] <= tokens_dynamic_cutoff) 
		)

		# List pages marked as likely covers
		list_of_covers = merged_expanded_df[merged_expanded_df['is_likely_cover']].page_number.unique().tolist()

		# Append frequencies and metadata
		volume_frequencies.append({
			'tokens_dominant_frequency': tokens_dominant_frequency,
			'tokens_positive_frequencies': tokens_positive_frequencies,
			'tokens_positive_amplitudes': tokens_positive_amplitudes,
			'digits_dominant_frequency': digits_dominant_frequency,
			'digits_positive_frequencies': digits_positive_frequencies,
			'digits_positive_amplitudes': digits_positive_amplitudes,
			'htid': merged_expanded_df['htid'].unique()[0],
			'lowercase_periodical_name': volume['lowercase_periodical_name'],
			'avg_tokens': merged_expanded_df['tokens_per_page'].mean(),
			'avg_digits': merged_expanded_df['digits_per_page'].mean(),
			'likely_covers': list_of_covers,
			'total_pages': merged_expanded_df['page_number'].nunique(),
			'total_tokens': merged_expanded_df['tokens_per_page'].sum(),
			'total_digits': merged_expanded_df['digits_per_page'].sum(),
			'table_row_index': volume['table_row_index'],
			'tokens_per_page': merged_expanded_df['tokens_per_page'],
			'page_numbers': merged_expanded_df['page_number'],
			'digits_per_page': merged_expanded_df['digits_per_page'],
			'tokens_dynamic_cutoff': tokens_dynamic_cutoff,
			'digits_dynamic_cutoff': digits_dynamic_cutoff,
			'missing_issues': missing_issues
		})

	# Create DataFrame from volume frequencies
	volume_frequencies_df = pd.DataFrame(volume_frequencies)
	
	if len(altair_charts) > 0:
		# Save Altair charts as images
		combined_charts = alt.vconcat(*altair_charts)
		# Save the chart
		save_chart(combined_charts, f"{output_dir}/annotated_tokens_per_page/{periodical_name}_tokens_per_page_chart.png", scale_factor=2.0)
	# Sequential correlation analysis (optional)
	if run_correlations:
		sequential_correlations = []
		for i in range(len(volume_frequencies) - 1):
			vol1 = volume_frequencies[i]
			vol2 = volume_frequencies[i + 1]
			
			# Interpolate or truncate amplitudes to match lengths if necessary
			min_length = min(len(vol1['tokens_positive_amplitudes']), len(vol2['tokens_positive_amplitudes']))
			corr, _ = pearsonr(
				vol1['tokens_positive_amplitudes'][:min_length], 
				vol2['tokens_positive_amplitudes'][:min_length]
			)
			sequential_correlations.append({
				'volume1': vol1['htid'], 
				'volume2': vol2['htid'], 
				'correlation': corr,
				'periodical_name': vol1['lowercase_periodical_name']
			})

		# Save sequential correlations as CSV
		sequential_corr_df = pd.DataFrame(sequential_correlations)
		sequential_corr_df.to_csv(f"{output_dir}/sequential_correlations.csv", index=False)

		# Plot frequency spectra for each volume
		for volume in volume_frequencies:
			plt.figure(figsize=(12, 6))
			plt.plot(volume['tokens_positive_frequencies'], volume['tokens_positive_amplitudes'], label='Tokens')
			plt.plot(volume['digits_positive_frequencies'], volume['digits_positive_amplitudes'], label='Digits')
			plt.title(f"Frequency Spectra - {volume['periodical_name']} (Volume {volume['volume_id']})")
			plt.xlabel('Frequency')
			plt.ylabel('Amplitude')
			plt.legend()
			volume_plot_path = f"{output_dir}/frequency_spectra_{volume['volume_id']}.png"
			plt.savefig(volume_plot_path)
			plt.close()  # Close the plot to save memory

	# Calculate consensus issue length based on median dominant frequency
	volume_frequencies_df['consensus_issue_length'] = volume_frequencies_df['tokens_dynamic_cutoff'].median()
	volume_frequencies_df['consensus_issue_length'] = volume_frequencies_df['consensus_issue_length'].fillna(0)

	plot_volume_frequencies_matplotlib(volume_frequencies, periodical_name, output_dir)
	plot_tokens_per_page(volume_frequencies, output_dir, periodical_name)

	return volume_frequencies_df

def generate_token_frequency_analysis(should_filter_greater_than_numbers: bool, should_filter_implied_zeroes: bool, should_run_correlations: bool, only_use_annotated_periodicals: bool, rerun_code: bool = False):
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

	volume_features_output_path = os.path.join("..", "datasets", "all_volume_features_and_frequencies.csv")
	volume_features_exist = False
	if os.path.exists(volume_features_output_path) and rerun_code:
		volume_features_df = read_csv_file(volume_features_output_path)
		volume_features_exist = True
		console.print(f"Found {len(volume_features_df)} existing volume features.", style="bright_green")
	elif os.path.exists(volume_features_output_path) and not rerun_code:
		#delete the file
		os.remove(volume_features_output_path)
		volume_features_df = pd.DataFrame()
	else:
		volume_features_df = pd.DataFrame()

	data_directory_path = get_data_directory_path()
	preidentified_periodicals_df = read_csv_file(os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", "periodical_metadata", "preidentified_periodicals_with_full_metadata.csv"))
	periodical_titles = preidentified_periodicals_df['lowercase_periodical_name'].unique()

	# Process only annotated periodicals if specified
	for index, title in enumerate(tqdm(periodical_titles, desc="Processing periodicals")):
		console.print(f"Processing periodical {title} number ({index + 1} out of {len(periodical_titles)})..", style="bright_blue")
		subset_preidentified_periodicals_df = preidentified_periodicals_df[preidentified_periodicals_df['lowercase_periodical_name'] == title]
		volumes = subset_preidentified_periodicals_df.volume_directory.unique()
		subset_matching_files_df = matching_files_df[matching_files_df['volume_directory'].isin(volumes)]

		# Skip unannotated periodicals if the flag is set
		if only_use_annotated_periodicals and len(subset_matching_files_df) == 0:
			console.print(f"No annotated files found for periodical {title}. Skipping...", style="bright_red")
			continue

		
		volume_paths = []

		for _, row in subset_preidentified_periodicals_df[subset_preidentified_periodicals_df.volume_directory.notna()].iterrows():
			if volume_features_exist:
				processed_htid = row.volume_directory.replace("_", ".")
				volume_in_features = volume_features_df[volume_features_df['htid'] == processed_htid].copy()
				if (len(volume_in_features) > 0) and (not rerun_code):
					console.print(f"Volume {row.volume_directory} already exists in volume features..", style="bright_yellow")
					continue

			matched_row = subset_matching_files_df[subset_matching_files_df['volume_directory'] == row.volume_directory]

			# Skip unannotated volumes if the flag is set
			if only_use_annotated_periodicals and len(matched_row) == 0:
				console.print(f"Volume {row.volume_directory} is not annotated. Skipping...", style="bright_red")
				continue
			
			is_annotated_periodical = len(matched_row) > 0
			file_path = matched_row.file_path.values[0] if len(matched_row) > 0 else os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", row.publication_directory, "volumes", row['volume_directory'], row['volume_directory'] + "_individual_tokens.csv")
			volume_paths.append({
				'file_path': file_path,
				'is_annotated_periodical': is_annotated_periodical,
				'should_filter_greater_than_numbers': should_filter_greater_than_numbers,
				'should_filter_implied_zeroes': should_filter_implied_zeroes,
				'table_row_index': row['table_row_index'],
				'lowercase_periodical_name': row['lowercase_periodical_name'],
				'htid': row['htid']
			})

		# If no volumes found, skip this periodical
		if len(volume_paths) == 0:
			console.print(f"No valid volumes found for periodical {title}. Skipping...", style="bright_red")
			continue

		volume_paths_df = pd.DataFrame(volume_paths)
		volume_frequencies = generate_volume_embeddings(volume_paths_df, output_dir="../figures", run_correlations=should_run_correlations)

		# Save volume frequencies to CSV
		if os.path.exists(volume_features_output_path):
			volume_frequencies.to_csv(volume_features_output_path, mode='a', index=False, header=False)
		else:
			volume_frequencies.to_csv(volume_features_output_path, index=False)

if __name__ == "__main__":
	filter_greater_than_numbers = True
	filter_implied_zeroes = True
	should_run_correlations = False
	should_rerun_code = False
	should_only_use_annotated_periodicals = False
	generate_token_frequency_analysis(filter_greater_than_numbers, filter_implied_zeroes, should_run_correlations, should_only_use_annotated_periodicals, should_rerun_code)