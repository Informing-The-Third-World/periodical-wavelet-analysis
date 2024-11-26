# Issue Segmentation Logic Outline

With a series of in-copyright or out-of-copyright Hathi Trust volumes, we should be able to use the following logic to segment the volumes into issues.

## Step 1: Volume Metadata 
The volume metadata should be used to determine the publication date of the volume. This will be used to order the issues. Also any subject metadata can be used to determine the publication frequency of the volume or likely genre of the publication. For example, a quarterly publication with the genre is likely to have 4 issues per year. 

Some potential headings: 
- `publication_date`
- `enumeration_chronology`
- `genre`

Headers from the metadata file:
```csv
issue_number,dates,page_number,type_of_page,original_volumes,notes,start_issue,end_issue,link,htid,original_source,record_url,periodical_name,publication_type,id,metadata_schema_version,enumeration_chronology,type_of_resource,title,date_created,pub_date,language,access_profile,isbn,issn,lccn,oclc,page_count,feature_schema_version,access_rights,alternate_title,category,genre_ld,genre,contributor_ld,contributor,handle_url,source_institution_ld,source_institution,lcc,type,is_part_of,last_rights_update_date,pub_place_ld,pub_place,main_entity_of_page,publisher_ld,publisher,lowercase_periodical_name,publication_directory,volume_directory
```

## Step 2: Tokens Per Page

This gives us a sense of the likely candidates for start/stop markers in the data. Also give info on how many images are in the volume, versus a more prose-based dataset. Use the tokens/character per page to determine the probability of a page being a cover, or a table of contents, or an advertisement - all info towards identifying the start/stop of an issue. This may not work for all languages (e.g. character number may not be a good indicator for Chinese). Effectively trying to use the length or amount of data in the dataset to determine the likelihood of a page being a certain type of page.

Identify the following:
- `tokens_per_page`
- `total_token_number`

## Step 3: Page Number Extraction

After subsetting to pages with lower thresholds of tokens per page, we can extract page numbers, eliminating all characters from the dataset that are non-numeric. We will keep track of pages even if they don't have numbers, as they may be useful for identifying the start/stop of an issue.

We can then also eliminate all numbers that are greater than the maximum page number in the volume. We can also use the implied differential between page numbers and digits to identify if a number is too large to be a page number. Both of these are somewhat interpretative, since there are journals that have page numbers in the thousands, but it is a good starting point.

## Page Number Sequence Alignment

Once we have the numbers on the page, we can align them to the actual page numbers in the volume. This will give us a sense of the likely start/stop of an issue. We can use sequence alignment to determine the most likely start/stop of an issue.

We are using the three following methods:

**1. Global Sequence Alignment (Needleman-Wunsch Method)**

This method applies the Needleman-Wunsch global sequence alignment algorithm to align the observed page numbers with a target sequence. The observed sequence is derived from the data, where missing values are replaced with a placeholder (-1). The alignment score is computed based on a customizable scoring matrix that assigns weights for matches, mismatches, and gaps.

Assumptions:

- The page numbers follow a linear progression (e.g., 1, 2, 3, etc.), which can serve as the target sequence.
- Alignment scores can effectively quantify the similarity between the observed and expected sequences, even when there are gaps or noise.
- The placeholder values (-1) appropriately represent missing or unrecognized digits without skewing the alignment.

**2. Probabilistic Detection with Sliding Windows**

This approach uses a sliding window mechanism to probabilistically detect the most likely sequence alignment. A sliding window iterates over subsets of the data, scoring the sequences based on metrics such as page range, implied differences, and the presence of non-digit pages. Cumulative scores are calculated for each window, and candidates exceeding a threshold score are identified as likely issues.

Assumptions:
- Smaller sections (windows) of the page number sequence can provide reliable clues about the overall sequence structure.
- Non-digit pages or missing values can be accommodated without significant impact on the detection of valid sequences.
- Probabilistic thresholds (e.g., score thresholds) can effectively differentiate valid issue sequences from random noise.

**3. Prefix Sum Calculation**

This method uses a prefix sum algorithm to calculate scores for potential issue boundaries across different threshold sizes and start pages. Raw scores are computed for each page number, considering digit presence and alignment with the target sequence. Prefix sums aggregate these scores across pages, and the configuration with the highest total score is identified as the best candidate.

Assumptions:
- Aggregated scores over a range of pages (prefix sums) can effectively capture the structure and alignment of issue boundaries.
- Configurations with the highest scores are likely to represent valid issues.
- The scoring matrix and weights (e.g., diagonal, up-down movements) accurately model the contributions of each page to the sequence.

*Shared Assumptions Across Methods*

- Page numbers are integral to determining sequence structure and issue boundaries.
- Missing or noisy data can be represented and managed using placeholders or probabilistic thresholds.
- Scores derived from alignment, probability, or aggregation are sufficient to identify the best candidates for issue detection.

These methods complement one another by combining deterministic alignment (global sequence), probabilistic inference (sliding window), and aggregate scoring (prefix sums), providing a robust framework for page number sequence alignment.

Currently we do use the confidence threshold from the first method to narrow the range of options for the second two, but we may remove that going forward. 

Once we have all three methods, we calculate scores for each method and combined the scores, and then use scikit-learn to assess the accuracy of the combined methods. We could calculate a score for each method, and then use a weighted average to determine the most likely start/stop of an issue.

## Token Likeliness on Page

We can use the actual tokens of the top identified pages to see if there are any patterns in the data that would help us identify the start/stop of an issue. For example, if there's multiple pages that have the name of the publication that's likely a cover, or if there's multiple pages that have the same table of contents, that's likely a table of contents.

To do this, we can use a simple count vectorizer to count the number of times a token appears on a page, and then use a cosine similarity to determine the similarity between pages. We can also use the textual features along with the previous steps in a Random Forest Classifier to determine the likelihood of a page being a cover, table of contents, or advertisement.
