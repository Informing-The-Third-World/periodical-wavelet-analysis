import pandas as pd
import os, sys
import glob
import warnings
warnings.filterwarnings('ignore')



def get_metadatas(dir, filename):
    cwd = os.getcwd()
    for subdir, dirs, files in os.walk(dir):
        for f in files:
            if filename in f:
                df = pd.read_csv(subdir+ '/'+f)
                print(f)
                print(df[0:1])
                vols = df.vol_id.tolist()
            
                # for vol in vols:
                #     command = f'htrc download -o {vol_dir} {vol}'
                #     os.system(command)

get_metadatas('./metadatas', 'liberator')