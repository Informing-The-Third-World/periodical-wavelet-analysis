import pandas as pd
from htrc_features import FeatureReader
import os
from tqdm import tqdm
# from pandarallel import pandarallel
import spacy

nlp = spacy.load('en_core_web_trf')
all_df = pd.read_csv(
    "../datsets/ht_ef_datasets/combined_full_hathitrust_annotated_magazines_with_htids.csv")
tqdm.pandas(desc="Spacy")
all_df['cleaned_text'] = all_df.token
all_df.cleaned_text = all_df.cleaned_text.fillna('')
all_df.cleaned_text = all_df.cleaned_text.astype(str)
all_df['cleaned_text'] = all_df.cleaned_text.str.encode(
    'ascii', 'ignore').str.decode('ascii')

# pandarallel.initialize(progress_bar=True)
all_df['cleaned_text'] = all_df.cleaned_text.progress_apply(lambda x: nlp(x))
all_df.to_csv(
    "../datsets/ht_ef_datasets/combined_full_hathitrust_annotated_magazines_with_htids_spacy_cleaned.csv", index=False)
