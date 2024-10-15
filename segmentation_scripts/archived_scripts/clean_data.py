import pandas as pd
import nltk
from nltk import word_tokenize
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

def process_dfs(return_length_min=100):
    arab_observer_df = pd.read_csv("../original_files/arab_observer_gv_processed.csv")
    tricontinental_bulletin_df = pd.read_csv(
        "../original_files/tricontinental_bulletin_gv_processed.csv")
    arab_observer_df = arab_observer_df[arab_observer_df.detected_language == 'en']
    tricontinental_bulletin_df = tricontinental_bulletin_df[tricontinental_bulletin_df.detected_language == 'en']

    arab_observer_df['text'] = arab_observer_df.text.fillna('')
    tricontinental_bulletin_df['text'] = tricontinental_bulletin_df.text.fillna('')
    arab_observer_df['text'] = arab_observer_df['text'].astype(str)
    tricontinental_bulletin_df['text'] = tricontinental_bulletin_df['text'].astype(
        str)
    arab_observer_df['text'] = arab_observer_df['text'].str.replace('\n', ' ')
    tricontinental_bulletin_df['text'] = tricontinental_bulletin_df['text'].str.replace(
        '\n', ' ')
    arab_observer_df['text'] = arab_observer_df['text'].str.lower()
    tricontinental_bulletin_df['text'] = tricontinental_bulletin_df['text'].str.lower(
    )
    tricontinental_bulletin_df = tricontinental_bulletin_df.rename(
        columns={'file_name': 'file_path', 'title': 'issue', 'page': 'page_number'})
    arab_observer_df['periodical_name'] = 'Arab Observer'
    tricontinental_bulletin_df['periodical_name'] = 'Tricontinental Bulletin'
    tricontinental_bulletin_df['date'] = tricontinental_bulletin_df.year

    dates = {
        'Date: 11/1968': '1968-11-01',
        'Date: 1/1969': '1969-01-01',
        'Date: 2/1969': '1969-02-01',
        'Date: 8/1969': '1969-08-01',
        'Date: 7/1971': '1971-07-01',
        'Date: 5/1971': '1971-05-01',
        'Date: 8/1968': '1968-08-01',
        'Date: 12/1971': '1971-12-01',
        'Date: 7/1968': '1968-07-01',
        'Date: 7/1972': '1972-07-01', 
        'Date: 1/1971': '1971-01-01',
        'Date: 2/1972': '1972-02-01',
        'Date: 11/1971': '1971-11-01',
        'Date: 6/1968': '1968-06-01',
        'Date: 9/1968': '1968-09-01',
        'Date: 10/1969': '1969-10-01',
        'Date: 12/1969': '1969-12-01', 
        'Date: 1/1970': '1970-01-01', 
        'Date: 2/1970': '1970-02-01', 
        'Date: 9/1971': '1971-09-01',
        'Date: 9/1972': '1972-09-01', 
        'Date: 4/1972': '1972-04-01', 
        'Date: 3/1969': '1969-03-01', 
        'Date: 3/1972': '1972-03-01',
        'Date: 5/1966': '1966-05-01', 
        'Date: 3/1970': '1970-03-01', 
        'Date: 5/1996': '1996-05-01',
        'Date: 1/1972': '1972-01-01',
    }
    exclude_dates = [ 'Year: 1999',  'Date: 5/1996', 'Year: 2004', 'Year: 2000']

    tricontinental_bulletin_df = tricontinental_bulletin_df[tricontinental_bulletin_df.year.isin(
        exclude_dates) == False]

    tricontinental_bulletin_df.date.replace(
        dates, inplace=True)

    tricontinental_bulletin_df['issue_number'] = tricontinental_bulletin_df.issue.str.extract(
        r'(\d+)')

    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1969') & (tricontinental_bulletin_df.issue_number == '11'), 'date'] = '1969-05-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1971') & (tricontinental_bulletin_df.issue_number == '21'), 'date'] = '1971-02-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1971') & (tricontinental_bulletin_df.issue_number == '25'), 'date'] = '1971-08-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1973') & (tricontinental_bulletin_df.issue_number == '33'), 'date'] = '1973-01-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date.isna()) & (tricontinental_bulletin_df.issue_number == '1'), 'date'] = '1966-04-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date.isna()) & (tricontinental_bulletin_df.issue_number == '3'), 'date'] = '1966-06-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date.isna()) & (tricontinental_bulletin_df.issue_number == '4'), 'date'] = '1966-07-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1968') & (tricontinental_bulletin_df.issue_number == '7'), 'date'] = '1968-01-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1969') & (tricontinental_bulletin_df.issue_number == '15'), 'date'] = '1968-07-01'

    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1967') & (tricontinental_bulletin_df.issue_number == '11'), 'date'] = '1967-02-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1967') & (tricontinental_bulletin_df.issue_number.isna()), 'date'] = '1967-04-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1969') & (tricontinental_bulletin_df.issue_number == '10'), 'date'] = '1969-04-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1970') & (tricontinental_bulletin_df.issue_number == '18'), 'date'] = '1969-12-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1970') & (tricontinental_bulletin_df.issue_number == '17'), 'date'] = '1969-11-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1971') & (tricontinental_bulletin_df.issue_number == '26'), 'date'] = '1971-08-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1971') & (tricontinental_bulletin_df.issue_number == '23'), 'date'] = '1971-04-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1972') & (tricontinental_bulletin_df.issue_number == '29'), 'date'] = '1972-05-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1972') & (tricontinental_bulletin_df.issue_number == '31'), 'date'] = '1972-06-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1972') & (tricontinental_bulletin_df.issue_number == '74'), 'date'] = '1972-12-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1973') & (tricontinental_bulletin_df.issue_number == '82'), 'date'] = '1973-02-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1973') & (tricontinental_bulletin_df.issue_number == '84'), 'date'] = '1973-04-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1975') & (tricontinental_bulletin_df.issue_number.isna()), 'date'] = '1975-01-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1976') & (tricontinental_bulletin_df.issue_number == '49'), 'date'] = '1976-01-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1976') & (tricontinental_bulletin_df.issue_number == '101'), 'date'] = '1976-12-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1977') & (tricontinental_bulletin_df.issue_number == '104'), 'date'] = '1977-01-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1977') & (tricontinental_bulletin_df.issue_number == '107'), 'date'] = '1977-12-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1980') & (tricontinental_bulletin_df.issue_number == '67'), 'date'] = '1980-01-01'
    tricontinental_bulletin_df.loc[(
        tricontinental_bulletin_df.date == 'Year: 1980') & (tricontinental_bulletin_df.issue_number == '69'), 'date'] = '1980-12-01'

    # subset_df = tricontinental_bulletin_df[tricontinental_bulletin_df.date.str.contains('Year', na=False)][['date', 'issue_number']].drop_duplicates()

    # subset_df['extracted_year'] = subset_df.date.str.extract(r'(\d+)')
    # subset_df['extracted_year'] = subset_df['extracted_year'].astype(int)
    # subset_df.sort_values(by=['extracted_year', 'issue_number'])

    arab_observer_df['cleaned_issue_date'] = pd.to_datetime(
        arab_observer_df['issue'], errors='coerce')

    tricontinental_bulletin_df['cleaned_issue_date'] = pd.to_datetime(tricontinental_bulletin_df['date'], errors='coerce')
    arab_observer_df['tokenized_text'] = arab_observer_df['text'].apply(lambda x: nltk.word_tokenize(x))
    tricontinental_bulletin_df['tokenized_text'] = tricontinental_bulletin_df['text'].apply(lambda x: nltk.word_tokenize(x))
    arab_observer_df['tokenized_length'] = arab_observer_df.tokenized_text.str.len()
    tricontinental_bulletin_df['tokenized_length'] = tricontinental_bulletin_df.tokenized_text.str.len()

    all_df = pd.read_csv("../ht_ef_datasets/combined_full_hathitrust_annotated_magazines_with_htids.csv")
    all_df['cleaned_issue_date'] = pd.to_datetime(all_df['start_issue'], errors='coerce')
    all_df = all_df.rename(columns={'sequence': 'page_number', 'token': 'text'})
    all_df.text = all_df.text.fillna('')
    tqdm.pandas(desc="tokenizing text")
    all_df['tokenized_text'] = all_df['text'].progress_apply(lambda x: nltk.word_tokenize(x))
    all_df['tokenized_length'] = all_df.tokenized_text.str.len()
    return arab_observer_df[arab_observer_df.tokenized_length > return_length_min], tricontinental_bulletin_df[tricontinental_bulletin_df.tokenized_length > return_length_min], all_df[all_df.tokenized_length >return_length_min]

