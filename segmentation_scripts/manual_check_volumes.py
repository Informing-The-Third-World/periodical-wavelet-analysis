from rich import print
from rich.console import Console
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


console = Console()
links_df = pd.read_csv(
    "../datasets/original_files/htrc_third_world_records_links_with_metadata.csv")

links_df["keep_volume"] = True


links_df = links_df.sort_values(by=['periodical_name', 'title'])
needs_checking = links_df.reset_index(drop=True)
names = links_df["periodical_name"].unique()
for name in names:
    print(f"Checking {name}")
    print(u'\u2500' * 10)
    for index, row in links_df[links_df["periodical_name"] == name].iterrows():
        print(f"On {index} out of {len(needs_checking)}")
        print(f"Checking named {row['periodical_name']} and title {row['title']}")
        print(f"Volume Link: {row['link']}")
        print(f"Record Link: {row['record_url']}")
        print(f"Publication type: {row['publication_type']}")
        print(f"Dates: {row['date']}")
        print(f"Access: {row['access']}")
        print(f"Rights: {row['rights']}")
        print(f"Description: {row['description']}")
        print(f"Source: {row['source']} and original source: {row['original_source']}")
        answer = console.input("Keep volume? [y/n] ")
        if answer == "n":
            links_df.loc[index, "keep_volume"] = False
            links_df.to_csv("../datasets/original_files/htrc_third_world_records_links_with_metadata.csv", index=False)
        print(u'\u2500' * 10)
for index, row in links_df.iterrows():
    print(f"On {index} out of {len(needs_checking)}")
    print(f"Checking named {row['periodical_name']} and title {row['title']}")
    print(f"Volume Link: {row['link']}")
    print(f"Record Link: {row['record_url']}")
    print(f"Publication type: {row['publication_type']}")
    print(f"Dates: {row['date']}")
    print(f"Access: {row['access']}")
    print(f"Rights: {row['rights']}")
    print(f"Description: {row['description']}")
    print(f"Source: {row['source']} and original source: {row['original_source']}")
    answer = console.input("Keep volume? [y/n] ")
    if answer == "n":
        links_df.loc[index, "keep_volume"] = False
        links_df.to_csv("../datasets/original_files/htrc_third_world_records_links_with_metadata.csv", index=False)
    print(u'\u2500' * 10)

links_df.to_csv("../datasets/original_files/htrc_third_world_records_links_with_metadata.csv", index=False)