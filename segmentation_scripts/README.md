# Scripts

1. HathiTrust Webscraper `webscrape_ht.py`
   - Uses the annotation data to scrape hathitrust for volumes
   - Creates the `annotation_metadata_mapping.csv` file
2. HathiTrust Extracted Features Cleaner `get_annotate_ht_volumes.py`
   - Creates the volumes from the extracted features and combines them
   - Creates the `directory_annotation_metadata_mapping.csv` file
3. Process HathiTrust Data `process_ht_volumes.py`
   - Processes and cleans the hathitrust text data with spaCy
   - Creates the final file `combined_full_hathitrust_annotated_magazines_with_htids_spacy_cleaned.csv`