import os
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()
import warnings
warnings.filterwarnings('ignore')
import sys

sys.path.append("..")
from load_datasets import *
# from compute_magazines.scripts.classifier_magazines import *

from tqdm import tqdm
from rich import print
from rich.console import Console

console = Console()

output_path = '../../datasets/ht_ef_datasets/full_hathitrust_annotated_magazines_with_htids.csv'
output_directory = "../../datasets/ht_ef_datasets/"
full_issues = get_full_combined_dataset(output_path, output_directory)

def clean_digits(df):
    df.token = df.token.astype(str)
    if 'volume_name' in df.columns.tolist():
        df.volume_number = df.volume_number.fillna(0)
    subset_digits = df[(df.token.str.isdigit()) ]
    subset_digits['number'] = subset_digits.token.astype(int)
    subset_digits['implied_zero'] = subset_digits.sequence.astype(int) - subset_digits.number
    return subset_digits

def get_predicted_page(df, range_of_max_issue_length):
    dfs = []
    for max_issue_length in range_of_max_issue_length:
        copied_df = df.copy()
        max_page = copied_df.sequence.max()
        max_possible_number = max_page + max_issue_length
        filtered_df = copied_df[copied_df.number < max_possible_number]
        grouped_pages = filtered_df.groupby('implied_zero').size().reset_index(name='counts')
        grouped_pages = grouped_pages[grouped_pages.counts > max_issue_length - 5]
        grouped_pages = grouped_pages.reset_index(drop=True)
        copied_df['predicted_page'] = None
        copied_df['predicted_issue_number'] = None
        copied_df['max_issue_length'] = max_issue_length
        
        for index, row in grouped_pages.iterrows():
            zero_window = [row.implied_zero -2, row.implied_zero + 2]
            zero_window = [max(0, x) for x in zero_window]
            for count in range(zero_window[0], zero_window[1]):
                copied_df.loc[(copied_df.sequence == count), 'predicted_page'] = 'predicted_beginning_of_issue'
                copied_df.loc[(copied_df.sequence == count), 'predicted_issue_number'] = f"issue {index}"
        dfs.append(copied_df)
    combined_df = pd.concat(dfs)
    return combined_df

### Annotated Datasets

def get_annotated_datasets(df, range_of_max_issue_length, rewrite_files):
       predictions_path = "../issue_segments/predicted_issue_segments_for_annotated_issues.csv"
       issues_path = "../derived_annotated_datasets/issue_segments_for_annotated_issues.csv"
       if (os.path.exists(predictions_path)) and (os.path.exists(issues_path)) and (rewrite_files == False):
              predictions = pd.read_csv(predictions_path)
              final_df = pd.read_csv(issues_path)
       else:
              subset_digits = clean_digits(df)
              tqdm.pandas()
              htids = subset_digits.htid.unique().tolist()
              dfs = []
              for htid in tqdm(htids, total=len(htids), desc='Processing digits'):
                     subset_df = subset_digits[subset_digits.htid == htid]
                     subset_df = get_predicted_page(subset_df, range_of_max_issue_length)
                     dfs.append(subset_df)
              combined_subset_digits = pd.concat(dfs)
              combined_subset_digits = combined_subset_digits.reset_index(drop=True)
              digit_cols = combined_subset_digits.columns.tolist()
              digit_cols = [x for x in digit_cols if x not in ['pos', 'count', 'section', 'token']]
              combined_subset_digits = combined_subset_digits[digit_cols]
              cols = list(set(df) & (set(combined_subset_digits)))
              

              merged_df = pd.merge(df, combined_subset_digits, on=cols, how='left')
              if merged_df['volume_number'].isna().all():
                     merged_df = merged_df.drop(columns='volume_number')
              cols = cols + ['predicted_page', 'predicted_issue_number', 'max_issue_length']
              subset_predictions = combined_subset_digits[cols].drop_duplicates()
              subset_predictions = subset_predictions[(subset_predictions.type_of_page == 'cover_page') & (subset_predictions.predicted_page.notna())]
              pages_cols = merged_df.columns.tolist()
              holdout_cols = ['section', 'token', 'pos', 'count', 'number', 'implied_zero']
              pages_cols = [x for x in pages_cols if x not in holdout_cols]
              pages_df = merged_df[pages_cols]
              pages_df = pages_df.drop_duplicates()
              groupby_cols = ['cleaned_magazine_title', 'ht_generated_title', 'volume_number', 'htid', 'hdl_link','cleaned_volume', 'start_issue', 'end_issue', 'datetime','dates', 'issue_number', 'type_of_page', 'sequence'] if 'volume_number' in merged_df.columns else ['cleaned_magazine_title', 'ht_generated_title', 'htid', 'hdl_link','cleaned_volume', 'start_issue', 'end_issue', 'datetime','dates', 'issue_number', 'type_of_page', 'sequence']
              final_df = merged_df.groupby(groupby_cols, as_index = False).agg({'token': ' '.join, 'count': list, 'number': list, 'implied_zero': list})
              final_cols = list(set(pages_df) & set(final_df))
              finalized_df = pd.merge(pages_df, final_df, on=final_cols, how='left')
              all_predictions = finalized_df[finalized_df.predicted_page.notna()]
              grouped_df = df.groupby(groupby_cols, as_index = False).agg({'token': ' '.join})
              pred_cols = list(set(grouped_df) & set(subset_predictions))

              pred_merged_df = pd.merge(grouped_df, subset_predictions, on=pred_cols, how='left')
              # predictions.to_csv(predictions_path, index=False)
              # finalized_df.to_csv(issues_path, index=False)
       return finalized_df, all_predictions, pred_merged_df

# full_issues.cleaned_magazine_title.value_counts()
# initial_df = full_issues[full_issues.htid == 'uiug.30112070958894']
existing_pubs = ['arab_observer_and_the_scribe', "tricontinental", "arab_observer"]
need_processing_pubs = full_issues[~full_issues.cleaned_magazine_title.isin(existing_pubs)].cleaned_magazine_title.unique().tolist()
for index, pub in enumerate(need_processing_pubs):
    subset_full_issues = full_issues[full_issues.cleaned_magazine_title == pub]
    htids = subset_full_issues.htid.unique().tolist()
    range_of_max_issue_length = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300]

    for index, htid in enumerate(htids):
        print(f"Processing {index} out of {len(htids)}")
        initial_df = subset_full_issues[subset_full_issues.htid == htid]
        annotated_df, all_predictions, predictions_df = get_annotated_datasets(initial_df, range_of_max_issue_length, rewrite_files=True)
        # annotated_dfs.append(annotated_df)
        # all_predictions_dfs.append(all_predictions)
        # predictions_dfs.append(predictions_df)
        htid = htid.replace('.', '')
        annotated_df.to_csv(f"../derived_files/temp/issue_segments_for_annotated_issues_{htid}.csv", index=False)
        all_predictions.to_csv(f"../derived_files/temp/predicted_issue_segments_for_annotated_issues_{htid}.csv", index=False)
        predictions_df.to_csv(f"../derived_files/temp/predicted_issue_segments_for_annotated_issues_merged_{htid}.csv", index=False)

# annotated_dfs = []
# all_predictions_dfs = []
# predictions_dfs = []
# for directory, subdir, files in os.walk("../derived_files/temp/"):
#     for file in files:
#         if file.startswith("issue_segments_for_annotated_issues"):
#             annotated_df = pd.read_csv(f"{directory}/{file}")
#             annotated_dfs.append(annotated_df)
#         elif file.startswith("predicted_issue_segments_for_annotated_issues"):
#             all_predictions = pd.read_csv(f"{directory}/{file}")
#             all_predictions_dfs.append(all_predictions)
#         elif file.startswith("predicted_issue_segments_for_annotated_issues_merged"):
#             predictions_df = pd.read_csv(f"{directory}/{file}")
#             predictions_dfs.append(predictions_df)
# final_annotated_df = pd.concat(annotated_dfs)
# final_all_predictions = pd.concat(all_predictions_dfs)
# final_predictions_df = pd.concat(predictions_dfs)
# print(len(final_annotated_df[(final_annotated_df.type_of_page == 'cover_page') & (final_annotated_df.predicted_page.notna())]), len(final_annotated_df[(final_annotated_df.type_of_page == 'cover_page') & (final_annotated_df.predicted_page.isna())]))

# final_annotated_df.to_csv("../derived_files/issue_segments_for_annotated_issues.csv", index=False)
# final_all_predictions.to_csv("../derived_files/predicted_issue_segments_for_annotated_issues.csv", index=False)
# final_predictions_df.to_csv("../derived_files/predicted_issue_segments_for_annotated_issues_merged.csv", index=False)