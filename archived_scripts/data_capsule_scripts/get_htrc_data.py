import pandas as pd
import os, sys
import glob
import nltk
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
import warnings
warnings.filterwarnings('ignore')
from progress.bar import IncrementalBar


def get_metadata_file(filename):
    cwd = os.getcwd()
    for subdir, dirs, files in os.walk('../data_capsule_scripts/metadatas'):
        for f in files:
            if filename in f:
                md = pd.read_csv(subdir+ '/'+f)
                return md

def get_hathi_files(dir, output_file, stopping, filename):
    '''Get all hathi text files and put them into csv for aggregating'''
    cwd = os.getcwd()

    md = get_metadata_file(filename)
    processing = IncrementalBar('processing text', max=len(md))

    for subdir, dirs, files in os.walk(dir):
        processing.next()
        os.chdir(dir)
        for f in files:
            if 'volume' not in f:
                page_number = int(f.split('.')[0])
                with open(subdir+ '/'+f, 'r') as file:
                    text = file.read()
                    df = {}
                    df['lowercase'] = text
                    df['page_number'] = page_number
                    df['vol_id'] = subdir.split('/')[-1]
                    row = md.loc[md['vol_id'] == df['vol_id']].copy()
                    df['title'] = filename
                    df['date_vol'] = row.date
                    d = df
                    df = pd.DataFrame().append(d, ignore_index=True)

                    os.chdir(cwd)
                    if os.path.exists(output_file):
                        df.to_csv(output_file, mode='a', header=False, index=False)
                    else:
                        df.to_csv(output_file, header=True, index=False)
    processing.finish()
        
if __name__ ==  "__main__" :
    direct = '/media/secure_volume/liberator/'
    output = 'liberator_test.csv'
    filename = 'liberator'
    get_hathi_files(direct, output, stopping, aggregating, liberator)
