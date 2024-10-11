
import os
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz
from htrc_features import FeatureReader
from datetime import datetime
from rich.console import Console

console = Console()

def transform_annotated_dates(rows):
	"""Transform metadata dates into mergeable dates"""
	date = rows[0:1].dates.values[0].replace('-', ' ').split(' ')
	day = date[1] if (len(date) == 3) and ('-' not in rows[0:1].dates.values[0]) else '1'
	start_month = date[0] 
	end_month = date[1] if (len(date) > 2) and ('-' in rows[0:1].dates.values[0]) else start_month
	year = date[-1]

	start_date = day +' ' + start_month + ' ' + year
	end_date = day +' ' + end_month + ' ' + year
	rows['start_issue'] = datetime.strptime(start_date, '%d %B %Y')
	rows['end_issue'] = datetime.strptime(end_date, '%d %B %Y')
	return rows

def cut_vols(rows):
	# Remove duplicates from issues and also make sure that sequence doesn't include first page for some reason I can't remember now...
	ends = rows[rows.type_of_page == 'end_of_issue'].sequence.tolist()[-1]
	rows = rows[rows.sequence <= ends]
	final_rows = rows
	# if any(rows.type_of_page == 'scanner_page') ==:
	first_page = rows[rows.type_of_page == 'cover_page'].sequence.values.tolist()
	if len(first_page) > 0:
		first_page = first_page[0]
	else:
		first_page = rows[rows.type_of_page == 'toc'].sequence.values.tolist()[0]
	final_rows = rows[rows.sequence > first_page -1]
	if any(rows.type_of_page == 'duplicates'):
		pages = rows[rows.type_of_page == 'duplicates'].notes.values[0]
		final_rows = rows[~rows.sequence.between(int(pages.split('-')[0]), int(pages.split('-')[1]))]
	return final_rows


def clean_annotated_df(annotated_df):
	"""Clean and normalize dates in the annotated datasets that were created manually in Notion"""
	annotated_df.columns = [x.lower().replace(' ', '_') for x in annotated_df.columns]
	annotated_df.notes = annotated_df['notes'].fillna('')
	annotated_df = annotated_df.fillna(method='ffill')
	
	annotated_df = annotated_df.groupby('dates').apply(transform_annotated_dates)
	return annotated_df
	

def merge_datasets(annotated_df, df):
	"""Merge extracted features dataset with the annotated one"""
	final_anno = df.merge(annotated_df, on=['original_volumes', 'sequence'], how='outer')
	final_anno = final_anno.sort_values(by=['original_volumes', 'sequence'])
	final_anno.type_of_page.fillna('content', inplace=True)

	final_anno.update(final_anno[['token','notes', 'section', 'pos']].fillna(''))
	final_anno.update(final_anno[['count']].fillna(0))
	final_anno.fillna(method='ffill', inplace=True)
	final_anno.fillna(method='bfill', inplace=True)
	# final_anno['implied_zero'] = final_anno['page_number'] - final_anno['number'] 
	
	# final_anno = final_anno.drop(columns=['index'])
	# final_anno = final_anno.drop_duplicates(subset=['date_vols', 'page_number'], keep='last')
	final_anno.reset_index(drop=True)
	# final_anno = final_anno.groupby('date', as_index=False).apply(cut_vols).reset_index()
	# final_anno = final_anno.loc[:, ~final_anno.columns.str.contains('^level')]
	return final_anno

def read_ids(md, folder, annotated_df):
	'''This function reads in the list of ids scraped from Hathi Trust and the folder destination. It gets the volume and tokenlist from Hathi Trust, and then calls spread table which separates out by page all tokens.'''
	directory = os.path.dirname(folder)
	if not os.path.exists(directory):
		os.makedirs(directory)

	volids = md['htid'].tolist()
	fr = FeatureReader(ids=volids)

	for vol in fr:
		row = md.loc[md['htid'] == vol.id].copy()
		# print(row, vol.title)
		title = vol.title if ':' not in vol.title else vol.title.split(':')[0]
		
		title = title.lower().replace('.', '').split(' ')
		magazine_title = "_".join(title)
		title = "_".join(title)+'_'+ '_'.join(str(row.date.values[0]).split(' '))
		title = title.replace(',', '_')
		title = title.replace('__', '_')
		file_name = folder+ '/' + title + '.csv'
		date_vols = row.date.values[0]
		subset_annotated_df = annotated_df.loc[annotated_df.original_volumes == date_vols]
		
		if not os.path.exists(file_name):
			# print(f'processing {file_name}')
			volume_df = vol.tokenlist(section='all')
			volume_df = volume_df.reset_index()
			file_name = file_name
			volume_df['magazine_title'] = magazine_title
			volume_df['title'] = title
			volume_df['htid'] = row.htid.values[0]
			volume_df['link'] = row.link.values[0]
			volume_df['original_volumes'] = date_vols
			volume_df = volume_df.rename(columns={'lowercase': 'token', 'page': 'sequence'})
			subset_annotated_df = subset_annotated_df.rename(columns={'page_number': 'sequence'})
			merged_df = merge_datasets(subset_annotated_df, volume_df)
			merged_df.to_csv(file_name, index=False)
		#     # spread_table(title, file_name) #Run this line if you want to group characters on pages into single rows
		# elif os.path.exists(file_name.split('.csv')[0] + '_grouped.csv'):
		#     add_volumes_dates(title, file_name, magazine_title, date_vols)
		# else:
		#     print(f'{file_name} already exists')

def add_volumes_dates(title, file_name, magazine_title, date_vols):
	output_file = file_name.split('.csv')[0] + '_grouped.csv'
	df = pd.read_csv(output_file)
	df['date_vols'] = date_vols
	df['magazine_title'] = magazine_title
	df['title'] = title

	df.to_csv(output_file, index=False)


def map_annotations_ht_directories(extracted_features_output_dir: str, rerun_code: bool) -> None:
	"""
	This function maps the extracted features directories to the annotated metadata files. It saves the mapping to a CSV file. It uses fuzzy matching to find the best match between the extracted features directories and the metadata files. If there is no match, it prompts the user to select the correct directory from a list.

	Args:
		extracted_features_output_dir (str): The directory containing the extracted features.
		rerun_code (bool): A boolean indicating whether to rerun the code.

	"""
	mapping_file_output_path = os.path.join('..', 'datasets', 'mapping_files', 'directory_annotation_metadata_mapping.csv')
	if os.path.exists(mapping_file_output_path) and rerun_code == False:
		console.print("Mapping file already exists. Skipping...", style="bold green")
		return
	else:
		# Get all directories of hathi_ef_datasets
		dir_list = [subdir.replace(extracted_features_output_dir, '') for subdir, _, _ in os.walk(extracted_features_output_dir)]
		subset_dir_list = [d for d in dir_list if d.count(os.sep) == 1]
		# Read the annotated mapping file
		annotated_mapping_output_path = os.path.join('..', 'datasets', 'mapping_files', 'annotation_metadata_mapping.csv')
		annotated_mapping_df = pd.read_csv(annotated_mapping_output_path)
		dfs = []

		# Iterate over all metadata files
		metadata_output_path = os.path.join('..', 'datasets', 'metadatas')
		for _, _, files in tqdm(os.walk(metadata_output_path)):
			for f in files:
				if f.endswith('.csv'):
					console.print(f'Processing {f}...', style="magenta")
					# Create file name and join directories for files
					file_name = f.rsplit('.', 1)[0]
					filenames = '_'.join([fi for fi in file_name.split('_') if not fi.isdigit()])
					appended_data = False
					final_dir = ''
					for dir_name in subset_dir_list:
						fuzziness = fuzz.ratio(filenames.replace('/', '').replace('_', ' '), dir_name.replace('/', '').replace('_', ' '))
						if fuzziness > 80:
							console.print(f"Found a match for {file_name} with {filenames.replace('/', '').replace('_', ' ')} and {dir_name.replace('/', '').replace('_', ' ')} of {fuzziness} fuzzy ratio.", style="bold green")
							appended_data = True
							final_dir = dir_name
							df = pd.DataFrame([{
								'local_dir': dir_name.split('/')[-1],
								'file_name': filenames,
								'fuzzy_ratio': fuzziness,
								'metadata_file': f,
								'final_dir': final_dir
							}])
							dfs.append(df)
							break
					if appended_data == False:
						# turn subdir list into enumerated dictionary
						processed_subset_dir_list = {i: d for i, d in enumerate(subset_dir_list)}
						console.print("No match found for", filenames.replace('/', '').replace('_', ' '), "in the subset directories. Please select the correct directory from the list below:", style="bold red")
						console.print(processed_subset_dir_list, style="bold blue")
						selected_dir = int(input("Please select the correct directory: "))
						final_dir = processed_subset_dir_list[selected_dir]
						df = pd.DataFrame([{
							'local_dir': final_dir.split('/')[-1],
							'file_name': filenames,
							'fuzzy_ratio': None,
							'metadata_file': f,
							'final_dir': final_dir
						}])
						dfs.append(df)

		# Combine and save the final DataFrame
		processed_df = pd.concat(dfs)
		final_df = pd.merge(annotated_mapping_df, processed_df, on='metadata_file')
		
		final_df.to_csv(mapping_file_output_path, index=False)
		console.print("Mapping file saved successfully.", style="bold green")

if __name__ == "__main__":
	# Define the relative path from the current working directory
	relative_extracted_features_path = os.path.join("..", "..", "..", "periodical-collection-curation", "HathiTrust-pcc-datasets", "datasets", "ht_ef_datasets")
	print(os.path.exists(relative_extracted_features_path))

	# Compute the absolute path
	absolute_extracted_features_path = os.path.abspath(relative_extracted_features_path)
	
	# Process the metadata files
	rerun_code = False
	map_annotations_ht_directories(absolute_extracted_features_path, rerun_code)
# Get relevant annotation file
					# annotation_row = annotated_mapping_df.loc[annotated_mapping_df['metadata_file'] == f].copy()
		#             # Clean and process annotated data
	#             annotated_df.Dates = annotated_df.Dates.str.replace('Decmeber', 'December')
	#             annotated_df.Dates = annotated_df.Dates.str.replace('Summer', 'July')
	#             annotated_df = clean_annotated_df(annotated_df)
	#             read_ids(md, final_dir, annotated_df)
	# Read metadata and annotation files
					# md = pd.read_csv(os.path.join(subdir, f), encoding="utf-8")
					# annotation_output_path = os.path.join("..", "datasets", "annotated_datasets", annotation_row.annotation_file.values[0])
					# annotated_df = pd.read_csv(annotation_output_path, encoding="utf-8")