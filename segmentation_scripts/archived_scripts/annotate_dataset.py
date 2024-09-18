import pandas as pd
from datetime import datetime
import numpy as np

def transform_annotated_dates(rows):
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
    final_rows = rows
    if any(rows.type_of_page == 'scanner_page') == False:
        first_page = rows[rows.type_of_page == 'cover_page'].page.values.tolist()[0]
        final_rows = rows[rows.page > first_page -1]
    if any(rows.type_of_page == 'duplicates'):
        pages = rows[rows.type_of_page == 'duplicates'].notes.values[0]
        final_rows = rows[~rows.page.between(int(pages.split('-')[0]), int(pages.split('-')[1]))]
    return final_rows


def clean_annotated_df(annotated_df):
    annotated_df.columns = [x.lower().replace(' ', '_') for x in annotated_df.columns]
    annotated_df.notes = annotated_df['notes'].fillna('')
    annotated_df = annotated_df.fillna(method='ffill')
    
    annotated_df = annotated_df.groupby('dates').apply(transform_annotated_dates)
    annotated_df = annotated_df.drop(columns=['dates'])
    return annotated_df

def merge_datasets(annotated_df, df):
    final_anno = df.merge(annotated_df, on=['date_vol', 'page_number'], how='outer')
    final_anno = final_anno.sort_values(by=['date_vol', 'page_number'])
    final_anno.type_of_page.fillna('content', inplace=True)

    final_anno.update(final_anno[['lowercase','notes']].fillna(''))
    final_anno.fillna(method='ffill', inplace=True)
    final_anno.fillna(method='bfill', inplace=True)
    
    final_anno = final_anno.drop(columns=['index'])
    final_anno = final_anno.drop_duplicates(subset=['date_vol', 'page_number'], keep='last')
    final_anno.reset_index(drop=True)
    final_anno = final_anno.groupby('date_vol', as_index=False).apply(cut_vols).reset_index()
    final_anno = final_anno.loc[:, ~final_anno.columns.str.contains('^level')]
    return final_anno


# '''Afro Asian Bulletin'''
# df = pd.read_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/Afro_Asian_Bulletin_1961_1967_HathiTrust/afro_asian_bulletin_1961_1967_volumes_processed_redo.csv')
# df.vols = df.vols.str.split('solidarity._').str.join('')
# annotated_df = pd.read_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/hathi_trust_annotated_datasets/afro_asian_bulletin_annotated_dataset.csv')
# annotated_df['vols'] = annotated_df['Original Volumes'].str.replace(',',' ').str.split(' ').str[:-1].str.join('_')
# annotated_df = clean_annotated_df(annotated_df)
# final_df = merge_datasets(annotated_df, df)
# final_df.to_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/Afro_Asian_Bulletin_1961_1967_HathiTrust/afro_asian_bulletin_1961_1967_annotated.csv')


# '''Liberator'''
# df = pd.read_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/Liberator_1961_1971_HathiTrust/redo/liberator_1961_1971_volumes_processed_redo.csv')
# annotated_df = pd.read_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/hathi_trust_annotated_datasets/liberator_annotated_dataset.csv')
# annotated_df['vols'] = np.where(annotated_df['Original Volumes'].str.replace(',',' ').str.split(' ').str.len() > 2, annotated_df['Original Volumes'].str.replace(',',' ').str.split(' ').str[0:2].str.join('_'), annotated_df['Original Volumes'].str.replace(',',' ').str.split(' ').str[0])
# annotated_df = clean_annotated_df(annotated_df)
# final_df = merge_datasets(annotated_df, df)
# final_df.to_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/Liberator_1961_1971_HathiTrust/liberator_1965_1971_annotated.csv')

'''Tricontinental'''
df = pd.read_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/Tricontinental_Bulletin_1966_1980_HathiTrust/redo/tricontinental_1961_1971_volumes_processed_redo.csv')
annotated_df = pd.read_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/hathi_trust_annotated_datasets/tricontinental_annotated_dataset.csv')
annotated_df['vols'] = annotated_df['Original Volumes'].str.replace('(',' ').str.replace(')', '').str.split(' ').str[0]
annotated_df = clean_annotated_df(annotated_df)
final_df = merge_datasets(annotated_df, df)
final_df.to_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/Tricontinental_Bulletin_1966_1980_HathiTrust/tricontinental_1966_1980_annotated.csv')

# '''The Scribe'''
# df = pd.read_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/the_scribe_1961_1965_HathiTrust/redo/the_scribe_1961_1965_volumes_processed_redo.csv')
# annotated_df = pd.read_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/hathi_trust_annotated_datasets/scribe_annotated_dataset.csv')
# annotated_df['vols'] = annotated_df['Original Volumes'].str.split(' ').str[:-1].str.join('_')
# annotated_df = clean_annotated_df(annotated_df)
# final_df = merge_datasets(annotated_df, df)
# final_df.to_csv('/Volumes/SecondDrive/dissertation/data_repos/data_sources/the_scribe_1961_1965_HathiTrust/the_scribe_1961_1965_annotated.csv')


