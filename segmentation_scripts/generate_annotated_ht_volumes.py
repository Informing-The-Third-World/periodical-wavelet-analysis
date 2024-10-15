import os
import pandas as pd
from tqdm import tqdm
from thefuzz import fuzz, process
from datetime import datetime
from rich.console import Console
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('..')
from segmentation_scripts.utils import read_csv_file, get_data_directory_path

console = Console()

def transform_annotated_dates(rows: pd.DataFrame) -> pd.DataFrame:
	"""Transform metadata dates into mergeable dates.
	
	Args:
		rows (pd.DataFrame): The rows to be transformed.
		
	Returns:
		pd.DataFrame: The transformed rows."""
	date = rows.iloc[0].dates.replace('-', ' ').split(' ')
	day = date[1] if (len(date) == 3) and ('-' not in rows.iloc[0].dates) else '1'
	start_month = date[0]
	end_month = date[1] if (len(date) > 2) and ('-' in rows.iloc[0].dates) else start_month
	year = date[-1]

	start_date = f"{day} {start_month} {year}"
	end_date = f"{day} {end_month} {year}"
	rows['start_issue'] = datetime.strptime(start_date, '%d %B %Y')
	rows['end_issue'] = datetime.strptime(end_date, '%d %B %Y')
	return rows

def clean_annotated_df(annotated_df: pd.DataFrame) -> pd.DataFrame:
	"""Clean and normalize dates in the annotated datasets that were created manually in Notion.
	
	Args:
		annotated_df (pd.DataFrame): The annotated dataset to be cleaned and normalized.
	
	Returns:
		pd.DataFrame: The cleaned and normalized annotated dataset."""
	annotated_df.columns = [x.lower().replace(' ', '_') for x in annotated_df.columns]
	annotated_df['notes'] = annotated_df['notes'].fillna('')
	annotated_df = annotated_df.ffill()
	annotated_df = annotated_df.groupby('dates', group_keys=False).apply(transform_annotated_dates, include_groups=True)
	return annotated_df

def cut_vols(rows: pd.DataFrame) -> pd.DataFrame:
	"""
	Filter volume rows based on the type of page and the presence of duplicates. Optional additional processsing of HT volumes to ensure that we are getting best issue definition as possible.

	Args:
		rows (pd.DataFrame): The rows to be filtered.

	Returns:
		pd.DataFrame: The filtered rows.
	"""
	# Make a copy to ensure original data is preserved
	final_rows = rows.copy()

	# Filter rows up to the last `end_of_issue` page_number, if present
	if 'end_of_issue' in final_rows['type_of_page'].values:
		end_page_number = final_rows[final_rows['type_of_page'] == 'end_of_issue']['page_number'].tolist()[-1]
		final_rows = final_rows[final_rows['page_number'] <= end_page_number]

	# Determine the first page_number to include based on `cover_page` or fallback to `toc`
	if 'cover_page' in final_rows['type_of_page'].values:
		first_page_number = final_rows[final_rows['type_of_page'] == 'cover_page']['page_number'].values[0]
	elif 'toc' in final_rows['type_of_page'].values:
		first_page_number = final_rows[final_rows['type_of_page'] == 'toc']['page_number'].values[0]
	else:
		first_page_number = 0  # Default to the beginning if neither `cover_page` nor `toc` is present

	# Exclude rows before the determined first page_number
	final_rows = final_rows[final_rows['page_number'] > first_page_number - 1]

	# Remove page_numbers within duplicate ranges, if `duplicates` is present
	if 'duplicates' in final_rows['type_of_page'].values:
		duplicates_range = final_rows[final_rows['type_of_page'] == 'duplicates']['notes'].values[0]
		start, end = map(int, duplicates_range.split('-'))
		final_rows = final_rows[~final_rows['page_number'].between(start, end)]

	return final_rows


def clean_annotated_ht_volume(annotated_ht_df:pd.DataFrame) -> pd.DataFrame:
	"""Process and clean an annotated HathiTrust volume.
	
	Args:
		annotated_ht_df (pd.DataFrame): The annotated HathiTrust volume to be processed.
	
	Returns:
		annotated_ht_df (pd.DataFrame): The processed annotated HathiTrust volume.
	"""
	# Fill forward to propagate the last non-null date through NaNs temporarily
	annotated_ht_df['temp_date'] = annotated_ht_df['dates'].ffill()

	# Create a unique group ID based on when a new date occurs
	annotated_ht_df['issue_id'] = (annotated_ht_df['temp_date'] != annotated_ht_df['temp_date'].shift()).cumsum()
	# Cast specific columns to desired types before filling
	annotated_ht_df['token'] = annotated_ht_df['token'].astype(str)
	annotated_ht_df['notes'] = annotated_ht_df['notes'].astype(str)
	annotated_ht_df['section'] = annotated_ht_df['section'].astype(str)
	annotated_ht_df['pos'] = annotated_ht_df['pos'].astype(str)
	annotated_ht_df['count'] = annotated_ht_df['count'].astype(float)   
	with pd.option_context('future.no_silent_downcasting', True):
		# Function to fill within each issue block
		def fill_issue_block(group):
			# Fill specific columns within the group
			group[['token', 'notes', 'section', 'pos']] = group[['token', 'notes', 'section', 'pos']].fillna('')
			group['count'] = group['count'].fillna(0)
			group['type_of_page'] = group['type_of_page'].fillna("content")
			
			# Forward-fill and backward-fill within the group
			group = group.ffill().bfill()
			
			return group

		# Apply the filling function to each group
		tqdm.pandas(desc="Filling issues")
		annotated_ht_df = annotated_ht_df.groupby('issue_id', group_keys=False).progress_apply(fill_issue_block)

	# Drop the temporary columns
	annotated_ht_df = annotated_ht_df.drop(columns=['temp_date', 'issue_id'])

	# Infer data types after filling
	# annotated_ht_df = annotated_ht_df.infer_objects()
	return annotated_ht_df

def merge_datasets(annotated_df: pd.DataFrame, preidentified_periodicals_df: pd.DataFrame, data_directory_path: str, cut_volumes: bool, rerun_code: bool, save_to_file: bool):
	"""Merge extracted features dataset with the annotated datasets.

	Args:
		annotated_df (pd.DataFrame): The annotated dataset to be merged.
		preidentified_periodicals_df (pd.DataFrame): The preidentified periodicals dataset to be merged.
		data_directory_path (str): The path to the data directory.
		cut_volumes (bool): A boolean indicating whether to cut volumes.
		rerun_code (bool): A boolean indicating whether to rerun the code.
		save_to_file (bool): A boolean indicating whether to save the merged dataset to a file.
	
	Returns:
		None
	"""
	periodical_name = preidentified_periodicals_df.lowercase_periodical_name.unique().tolist()[0]
	merged_annotated_file_path = os.path.join("..", "datasets", "metadatas", f"{periodical_name}_merged_annotated_preidentified_periodical_with_full_metadata.csv")
	if os.path.exists(merged_annotated_file_path) and rerun_code == False:
		console.print("Merged annotated file already exists. Skipping...", style="bold green")
		merged_annotated_df = read_csv_file(merged_annotated_file_path)
	else:
		console.print(f"Processing {periodical_name}...", style="bold blue")
		annotated_df.Dates = annotated_df.Dates.str.replace('Decmeber', 'December')
		annotated_df.Dates = annotated_df.Dates.str.replace('Summer', 'July')
		cleaned_annotated_df = clean_annotated_df(annotated_df)
		preidentified_periodicals_df = preidentified_periodicals_df.rename(columns={'date': 'original_volumes'})
		# Merge and subset to only the volumes that are in the annotated dataset
		merged_annotated_df = pd.merge(cleaned_annotated_df, preidentified_periodicals_df, on='original_volumes', how='inner')
		if save_to_file:
			merged_annotated_df.to_csv(merged_annotated_file_path, index=False)
		console.print(f"Initially we had {preidentified_periodicals_df.htid.nunique()} HathiTrust volumes, but after merging with the annotated dataset, we have {merged_annotated_df.htid.nunique()} volumes.", style="bold green")

	htids_list = merged_annotated_df.htid.unique().tolist()
	for htid_value in htids_list:
		console.print(f"Processing {htid_value}...", style="bold blue")
		subset_merged_annotated_df = merged_annotated_df[merged_annotated_df.htid==htid_value]
		publication_directory = subset_merged_annotated_df.publication_directory.values[0]
		new_publication_directory = subset_merged_annotated_df.lowercase_periodical_name.values[0]
		volume_directory = subset_merged_annotated_df.volume_directory.values[0]
		annotated_ht_directory = os.path.join("..", "datasets", "annotated_ht_ef_datasets", new_publication_directory, volume_directory)
		if not os.path.exists(annotated_ht_directory):
			os.makedirs(annotated_ht_directory, exist_ok=True)
		annotated_ht_file_path = os.path.join(annotated_ht_directory, f"{htid_value.replace('.', '_')}_annotated_individual_tokens.csv")		
		if os.path.exists(annotated_ht_file_path) and rerun_code == False:
			console.print(f"Annotated HathiTrust file for {htid_value} already exists. Skipping...", style="bold green")
		else:
			ht_file_path = os.path.join(data_directory_path, "HathiTrust-pcc-datasets", publication_directory, volume_directory, f"{htid_value.replace('.', '_')}_individual_tokens.csv")
			if os.path.exists(ht_file_path):
				ht_df = read_csv_file(ht_file_path)
				ht_df = ht_df.rename(columns={'page': 'page_number', 'periodical_name': 'lowercase_periodical_name'})
				annotated_ht_df = pd.merge(subset_merged_annotated_df, ht_df, on=['lowercase_periodical_name', 'htid', 'record_url', 'page_number'], how='outer')
				processed_annotated_ht_df = clean_annotated_ht_volume(annotated_ht_df)
				processed_annotated_ht_df = processed_annotated_ht_df.sort_values(by=['original_volumes', 'page_number'])
				
				tqdm.pandas(desc="Cutting volumes")
				cut_annotated_ht_df = processed_annotated_ht_df.groupby('dates', group_keys=False).progress_apply(cut_vols, include_groups=True) if cut_volumes else processed_annotated_ht_df

				cut_annotated_ht_df = cut_annotated_ht_df.reset_index(drop=True)
				cut_annotated_ht_df = cut_annotated_ht_df.loc[:, ~cut_annotated_ht_df.columns.str.contains('^level')]
				if save_to_file:
					cut_annotated_ht_df.to_csv(annotated_ht_file_path, index=False)
			else:
				console.print(f"File {ht_file_path} does not exist. Skipping...", style="bold red")
				continue

def map_annotated_ht_volumes( data_directory_path: str, rerun_code: bool, cut_volumes:bool, save_to_file:bool) -> None:
	"""
	This function maps the annotated HathiTrust volumes to the preidentified periodicals with full metadata. It merges the datasets and saves the merged dataset to a file.

	Args:
		data_directory_path (str): The path to the data directory.
		rerun_code (bool): A boolean indicating whether to rerun the code.
		cut_volumes (bool): A boolean indicating whether to cut volumes.
		save_to_file (bool): A boolean indicating whether to save the merged dataset to a file.
	
	Returns:
		None
	"""
	
	preidentified_periodicals_df = read_csv_file(os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", "preidentified_periodicals_with_full_metadata.csv"))
	preidentified_periodical_titles = preidentified_periodicals_df['lowercase_periodical_name'].unique().tolist()

	# Iterate over all metadata files
	annotated_dataset_output_path = os.path.join('..', 'datasets', 'annotated_datasets')
	for _, _, files in tqdm(os.walk(annotated_dataset_output_path)):
		for annotated_dataset_file_path in files:
			if annotated_dataset_file_path.endswith('.csv'):
				console.print(f'Processing {annotated_dataset_file_path}...', style="magenta")
				# Create file name and join directories for files
				file_name = annotated_dataset_file_path.rsplit('.', 1)[0].replace('_annotated_dataset', '')
				filenames = '_'.join([fi for fi in file_name.split('_') if not fi.isdigit()])
				
				fuzzy_matches = process.extract(filenames, preidentified_periodical_titles, scorer=fuzz.token_sort_ratio, limit=5)
				top_match, match_score = fuzzy_matches[0]

				if match_score == 100:
					subset_df = preidentified_periodicals_df[preidentified_periodicals_df['lowercase_periodical_name'] == top_match]
				else:
					# Show options if no perfect match
					console.print(f"No exact match for {file_name}. Here are the closest matches:", style="bold red")
					for idx, (title, score) in enumerate(fuzzy_matches):
						console.print(f"[{idx}] {title} (Score: {score})", style="bold blue")
						possible_df = preidentified_periodicals_df[preidentified_periodicals_df['lowercase_periodical_name'] == title]
						console.print(possible_df[['lowercase_periodical_name', 'htid', 'record_url']].to_string(index=False), style="bold green")
					
					# Let the user select an option
					selected_idx = int(input("Select the correct match by entering the index number: "))
					correct_title = fuzzy_matches[selected_idx][0]
					subset_df = preidentified_periodicals_df[preidentified_periodicals_df['lowercase_periodical_name'] == correct_title]
				
				# Read and process the annotated file
				annotated_df = read_csv_file(os.path.join(annotated_dataset_output_path, annotated_dataset_file_path))
				if "Original Volumes" not in annotated_df.columns:
					console.print(f"Original Volumes column not found in {annotated_dataset_file_path}. Skipping...", style="bold red")
					continue
				merge_datasets(annotated_df, subset_df, data_directory_path, cut_volumes, rerun_code, save_to_file)

if __name__ == "__main__":
	data_directory_path = get_data_directory_path()
	should_rerun_code = False
	should_cut_volumes = True
	should_save_to_file = True
	map_annotated_ht_volumes(data_directory_path, should_rerun_code, should_cut_volumes, should_save_to_file)
