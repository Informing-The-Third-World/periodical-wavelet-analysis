import os
import pandas as pd
from typing import List, Optional
import apikey
import altair as alt
import vl_convert as vlc

from rich.console import Console
from rich.table import Table
import sys
import re
from tqdm import tqdm

console = Console()

def save_chart(chart: alt.Chart, filename: str, scale_factor=2.0) -> None:
	'''
	Save an Altair chart using vl-convert
	
	Parameters
	----------
	chart : alt.Chart
		The Altair chart to save.
	filename: str
		The filename to save the chart as.
	scale_factor: float, optional
		The factor to scale the image resolution by.
		E.g. A value of `2` means two times the default resolution.

	Returns
	-------
	None
	'''
	with alt.data_transformers.enable("default"), alt.data_transformers.disable_max_rows():
		if filename.split('.')[-1] == 'svg':
			with open(filename, "w") as f:
				f.write(vlc.vegalite_to_svg(chart.to_dict()))
		elif filename.split('.')[-1] == 'png':
			with open(filename, "wb") as f:
				f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
		else:
			raise ValueError("Only svg and png formats are supported")

def set_data_directory_path(path: str) -> None:
	"""
	Sets data directory path.

	Parameters
	----------
	path : str
		The path to set as the data directory.

	Returns
	-------
	None
	"""
	apikey.save("ITTW_DATA_DIRECTORY_PATH", path)
	console.print(f'Informing the Third World data directory path set to {path}', style='bold blue')

def get_data_directory_path() -> str:
	"""
	Gets data directory path.

	Parameters
	----------
	None
	
	Returns
	-------
	str
		The data directory path.
	"""
	return apikey.load("ITTW_DATA_DIRECTORY_PATH")

def read_csv_file(file_name: str, directory: Optional[str] = None, encodings: Optional[List[str]] = None, error_bad_lines: Optional[bool] = False) -> Optional[pd.DataFrame]:
	"""
	Reads a CSV file into a pandas DataFrame. This function allows specification of the directory, encodings, and handling of bad lines in the CSV file. If the file cannot be read, the function returns None.

	Parameters
	----------
	file_name: str
		The name of the CSV file to read.
	directory: Optional[str]
		Optional string specifying the directory where the file is located. If None, it is assumed the file is in the current working directory.
	encodings: Optional[List[str]]
		Optional list of strings specifying the encodings to try. Defaults to ['utf-8'].
	error_bad_lines: Optional[bool]
		Optional boolean indicating whether to skip bad lines in the CSV. If False, an error is raised for bad lines. Defaults to False.

	Returns
	-------
	Optional[pd.DataFrame]
		The DataFrame containing the CSV data, or None if the file could not be read.
	"""
	# Set default encodings if none are provided
	if encodings is None:
		encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-8-sig']

	# Read in the file
	file_path = file_name if directory is None else os.path.join(directory, file_name)
	
	# Try to read the file with each encoding
	for encoding in encodings:
		try:
			# Return the dataframe
			return pd.read_csv(file_path, low_memory=False, encoding=encoding, on_bad_lines='warn' if error_bad_lines else 'error')
		# If there's a Pandas error, print it and return None
		except pd.errors.EmptyDataError:
			console.print(f'Empty dataframe for {file_name}', style='bold red. Printed from function read_csv_file.')
			return None
		# If there's an encoding error, print it and try the next encoding
		except UnicodeDecodeError:
			console.print(f'Failed to read {file_name} with {encoding} encoding. Trying next encoding... Printed from function read_csv_file.', style='bold yellow')
		# If there's another type of error, print it and return None
		except Exception as e:
			console.print(f'Failed to read {file_name} with {encoding} encoding. Error: {e}. Printed from function read_csv_file.', style='bold red')
			return None

	# If none of the encodings worked, print an error and return None
	console.print(f'Failed to read {file_name} with any encoding. Printed from function read_csv_file.', style='bold red')
	return None

def generate_table(df: pd.DataFrame, table_title: str) -> None:
	"""
	Given a DataFrame, generate a Rich Table and print it to the console.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame to be printed.
	table_title : str
		The title of the table.

	Returns
	-------
	None
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
	"""
	Clean and filter digit tokens in the DataFrame while retaining non-digit pages.
	
	Parameters
	----------
	df : pd.DataFrame
		The DataFrame to be cleaned.
	filter_greater_than_numbers : bool
		Whether to filter out digits greater than the maximum page number.
	filter_implied_zeroes : bool
		Whether to filter out implied zeroes.
	preidentified_periodical : bool
		Whether the DataFrame contains preidentified periodical data.
		
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

def process_file(file_path: str, is_preidentified_periodical: bool, should_filter_greater_than_numbers: bool, should_filter_implied_zeroes: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Process a CSV file to clean and filter digit tokens while retaining non-digit pages.

	Parameters
	----------
	file_path : str
		The path to the CSV file to be processed.
	is_preidentified_periodical : bool
		Whether the file is a preidentified periodical.
	should_filter_greater_than_numbers : bool
		Whether to filter out digits greater than the maximum page number.
	should_filter_implied_zeroes : bool
		Whether to filter out implied zeroes.

	Returns
	-------
	tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
		A tuple containing the expanded DataFrame, the cleaned subset of digits, and the grouped DataFrame.
	"""
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

if __name__ == '__main__':
	set_data_directory_path('/Users/zleblanc/Informing-The-Third-World/periodical-collection-curation')
	data_directory_path = get_data_directory_path()
	console.print(f'Data directory path: {data_directory_path}', style='bold blue')