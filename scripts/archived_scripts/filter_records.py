import os
import shutil
from tqdm import tqdm
import pandas as pd
from typing import Dict, List

def extract_record_url(records: List[str]) -> Dict[str, str]:
    """
    Extracts the record URL from a list of records.

    Each record is a string, and the function searches for a numeric ID within each record.
    It then constructs a URL based on this ID and returns a dictionary mapping the original
    record to the updated URL and the record ID.

    Args:
        records (List[str]): A list of record strings.

    Returns:
        Dict[str, str]: A dictionary with keys 'record_url', 'updated_record_url', and 'record_id',
                        and corresponding values from the input records.
    """
    records_mapping = {}
    for record in records:
        for chunk in record.split(' '): 
            if chunk.isdigit():
                record_id = chunk
                url = f'https://catalog.hathitrust.org/Record/{record_id}'
                records_mapping['record_url'] = record
                records_mapping['updated_record_url'] = url
                records_mapping['record_id'] = record_id
                return records_mapping

def filter_records(existing_records: str, annotation_directory: str) -> None:
    """
    Filters records based on existing records and annotations.

    This function reads existing records and annotations from CSV files, processes them,
    and writes the filtered records to new CSV files.

    Args:
        existing_records (str): Path to the CSV file containing existing records.
        annotation_directory (str): Path to the directory containing annotation CSV files.

    Returns:
        None
    """
    output_file = 'annotation_metadata_mapping.csv'
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists('../datasets/metadatas'):
        shutil.rmtree('../datasets/metadatas')
    os.makedirs('../datasets/metadatas')
    existing_df = pd.read_csv(existing_records)
    for file in tqdm(os.listdir(annotation_directory), desc='Filtering records'):
        if (file.endswith('.csv')) and ('freedomways' not in file):
            annotation_df = pd.read_csv(os.path.join(annotation_directory, file), encoding='utf-8')
            annotation_df.columns = ['_'.join(x.lower().split(' ')) for x in annotation_df.columns]
            annotation_df.notes.fillna('', inplace=True)
            annotation_df['record_url'] = None
            annotation_df.loc[annotation_df.notes.str.lower().str.contains('record'), 'record_url'] = annotation_df.loc[annotation_df.notes.str.lower().str.contains('record'), 'notes']
            annotation_df.record_url = annotation_df.record_url.ffill()
            records = annotation_df.record_url.unique().tolist()
            records_mapping = extract_record_url(records)
            annotation_df['updated_record_url'] = annotation_df.record_url.map(records_mapping)
            annotation_df['record_id'] = annotation_df.updated_record_url.map(records_mapping)
            annotation_df = annotation_df.drop(columns=['record_url'])
            annotation_df = annotation_df.rename(columns={'updated_record_url': 'record_url'})
            subset_existing_df = existing_df[existing_df['record_url'].isin(annotation_df.record_url.unique())]
            record_ids = '_'.join(subset_existing_df.record_url.unique().tolist())
            new_file_name = file.replace('_annotated_dataset.csv', f'_{record_ids}.csv')
            output_path = os.path.join('../datasets/metadatas', new_file_name)
            subset_existing_df.to_csv(output_path, index=False)
            df = pd.DataFrame([{'annotation_file': os.path.join(annotation_directory, file), 'metadata_file': output_path, 'magazine_name': file.split('_annotated')[0]}])
            if os.path.exists(output_file):
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, header=True, index=False)

if __name__ == '__main__':
    existing_records = '../periodical_curation_collection/ht_periodicals/datasets/preidentified_periodicals_with_metadata.csv'
    annotation_directory = '../datasets/annotations'
    filter_records(existing_records, annotation_directory)
