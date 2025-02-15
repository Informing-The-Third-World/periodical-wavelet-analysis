# Scripts Folder

*Currently a work-in-progress*

This folder contains a variety of scripts for periodical issue segmentation and modeling. There are currently three subfolders:

- **`archived_scripts`**: Contains previously used scripts that are no longer in use.
- **`segmentation_scripts`**: Contains scripts for periodical issue segmentation.
- **`wavelet_scripts`**: Contains scripts for wavelet modeling.

In this main folder, we have two scripts:

- **`utils.py`**: Contains utility functions used by other scripts in this project. Below is a detailed breakdown of the functions included:

| Function                | Description                                                                                     | Parameters                                                                                                                                                                                                                                                                                                                                                                             | Returns                                    |
|-------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| `save_chart`            | Saves an Altair chart as a `.png` or `.svg` file.                                               | `chart` (Altair chart), `filename` (str, path to save), `scale_factor` (float, resolution scaling factor, default `2.0`)                                                                                                                                                                                                                         | None                                       |
| `set_data_directory_path` | Sets a path to the data directory and saves it as a persistent variable.                       | `path` (str, directory path to set)                                                                                                                                                                                                                                                                                                             | None                                       |
| `get_data_directory_path` | Retrieves the currently set data directory path.                                              | None                                                                                                                                                                                                                                                                                                                                           | `str` (directory path)                     |
| `read_csv_file`         | Reads a CSV file into a Pandas DataFrame with support for multiple encodings and error handling. | `file_name` (str, CSV file name), `directory` (Optional[str], directory path, default `None`), `encodings` (Optional[List[str]], list of encodings, default `['utf-8', 'latin1', 'iso-8859-1', 'utf-8-sig']`), `error_bad_lines` (bool, skip bad lines, default `False`)                                                                           | Pandas DataFrame or `None`                 |
| `generate_table`        | Displays a Pandas DataFrame as a styled table in the console.                                   | `df` (Pandas DataFrame), `table_title` (str, title for the table)                                                                                                                                                                                                                                                                               | None                                       |
| `filter_integers`       | Checks if a given string token represents an integer.                                           | `token` (str, token to check)                                                                                                                                                                                                                                                                                                                   | `bool` (`True` if integer, else `False`)   |
| `calculate_digit_coverage` | Calculates the number of digits in rows of a DataFrame.                                      | `rows` (Pandas DataFrame, with column `implied_zero`)                                                                                                                                                                                                                                                                                          | `int` (number of digits)                   |
| `clean_digits`          | Cleans and filters digit tokens while retaining non-digit pages.                                | `df` (Pandas DataFrame), `filter_greater_than_numbers` (bool), `filter_implied_zeroes` (bool), `preidentified_periodical` (bool)                                                                                                                                                                                                                 | Cleaned Pandas DataFrame                   |
| `process_file`          | Handles file reading, token expansion, cleaning, while retaining file pages for issue segmentation.                        | `file_path` (str, path to CSV), `is_preidentified_periodical` (bool), `should_filter_greater_than_numbers` (bool), `should_filter_implied_zeroes` (bool)                                                                                                                                                                                        | Expanded DataFrame, digit subset, grouping |


- **`generate_annotated_ht_volumes.py`**: A script for combining annotated volumes with the original HT volumes and processing them for segmentation. Below is a detailed breakdown of its functionality:

| Function                   | Description                                                                                                                                                               | Parameters                                                                                                                                                                                                                                                                                                       | Returns                     |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| `transform_annotated_dates` | Transforms metadata dates into a format that can be merged.                                                                                                              | `rows` (pd.DataFrame): DataFrame rows to transform.                                                                                                                                                                                                                                                              | Transformed `pd.DataFrame`. |
| `clean_annotated_df`       | Cleans and normalizes dates in manually annotated datasets.                                                                                                               | `annotated_df` (pd.DataFrame): Annotated dataset to clean and normalize.                                                                                                                                                                                                                                         | Cleaned `pd.DataFrame`.     |
| `cut_vols`                 | Filters volume rows based on page type and removes duplicates for optimal issue segmentation.                                                                              | `rows` (pd.DataFrame): Rows to filter.                                                                                                                                                                                                                                                                           | Filtered `pd.DataFrame`.    |
| `clean_arab_observer_df`   | Cleans and normalizes dates in the Arab Observer dataset.                                                                                                                 | `arabobserver_df` (pd.DataFrame): Dataset to clean and normalize.                                                                                                                                                                                                                                                | Cleaned `pd.DataFrame`.     |
| `clean_afro_asian_df`      | Cleans and normalizes dates in the Afro-Asian dataset.                                                                                                                    | `afroasian_df` (pd.DataFrame): Dataset to clean and normalize.                                                                                                                                                                                                                                                   | Cleaned `pd.DataFrame`.     |
| `clean_annotated_ht_volume` | Processes and cleans an annotated HathiTrust volume.                                                                                                                     | `annotated_ht_df` (pd.DataFrame): Annotated HathiTrust volume to clean. <br> `volume_type` (str): Type of volume, either `grouped` or `individual`.                                                                                                                                                              | Cleaned `pd.DataFrame`.     |
| `merge_datasets`           | Merges extracted feature datasets with annotated datasets and optionally saves the merged dataset.                                                                         | `annotated_df` (pd.DataFrame): Annotated dataset. <br> `preidentified_periodicals_df` (pd.DataFrame): Preidentified periodicals dataset. <br> `data_directory_path` (str): Path to data directory. <br> `cut_volumes` (bool): Whether to filter rows. <br> `rerun_code` (bool): Whether to rerun processing. <br> `save_to_file` (bool): Whether to save the dataset. <br> `volume_type` (str): Type of volume (`grouped` or `individual`). | None.                      |
| `map_annotated_ht_volumes` | Maps annotated HathiTrust volumes to preidentified periodicals, merges datasets, and processes volumes with optional filtering and saving.                                  | `data_directory_path` (str): Path to the data directory. <br> `rerun_code` (bool): Whether to rerun processing. <br> `cut_volumes` (bool): Whether to filter volumes. <br> `save_to_file` (bool): Whether to save processed datasets. <br> `volume_type` (str): Volume type (`grouped` or `individual`).                         | None.                      |

# Serial Token Frequency As Wavelet & Signal Processing Analysis Scripts Notes

This folder contains the code and documentation for analyzing token frequency data extracted from OCR text as a signal, enabling the use of signal processing techniques to extract meaningful patterns. Essentially, it treats token frequency as a one dimensional waveform, allowing us to apply signal processing techniques to extract meaningful patterns and trends.

## Core Concepts & Assumptions

Metrics to Include

These metrics directly evaluate the quality of the wavelet representation or its alignment with the original signal:

*Reconstruction Quality*

1. wavelet_mse (Mean Squared Error):

   - Why Include: A lower MSE indicates better reconstruction fidelity. It’s a core measure of how well the wavelet transform approximates the original signal.

2. wavelet_psnr (Peak Signal-to-Noise Ratio):

   - Why Include: Complements MSE by quantifying reconstruction quality on a logarithmic scale. It’s particularly useful when MSE alone doesn’t fully capture perceptual quality.

*Efficiency & Compactness*

3. wavelet_sparsity:

   - Why Include: Reflects the compactness of the wavelet representation, which can be valuable for identifying efficient wavelet transforms.

*Statistical & Structural Fidelity*

4. wavelet_energy_entropy:

   - Why Include: Indicates the balance between energy distribution and entropy, highlighting how well the wavelet preserves signal structure. Particularly relevant for token frequency signals, which may have structural patterns.

5. emd_value (Earth Mover’s Distance):

   - Why Include: Measures distributional differences between the original and reconstructed signals, providing insight into alignment beyond pixel-wise errors.

6. kl_divergence:

   - Why Include: Captures information-theoretic differences between the original and reconstructed signals. Useful for evaluating how well the transform preserves statistical characteristics.

*Signal Integrity*

7. smoothness:

    - Why Include: Highlights the degree to which the reconstructed signal avoids oscillatory noise. This can be valuable if smoothness is desired in your analysis.

8. correlation:

   - Why Include: Measures the linear relationship between the original and reconstructed signals, which is crucial for preserving signal integrity.

*Multi-Scale Analysis*

9. avg_variance_across_levels:

   - Why Include: Offers a holistic view of energy distribution across wavelet decomposition levels, which can be critical for capturing the signal’s multi-scale characteristics.

Metrics to Exclude

These metrics are better treated as informational rather than included in rankings:

1. wavelet_adaptive_threshold:

   - Why Exclude: It’s primarily descriptive of the sparsity threshold derived for the wavelet coefficients. While informative, it doesn’t directly evaluate performance.

2. signal_length:

   - Why Exclude: This is an intrinsic property of the signal and doesn’t contribute to evaluating wavelet performance.

3. decomposition_levels / scales_used:

   - Why Exclude: While these can influence performance, they are better used to describe configurations rather than directly ranking wavelets.

4. variance_ratio_across_levels:

   - Why Exclude: While potentially useful for diagnostics, its relevance to ranking is ambiguous. A high ratio could signal good performance or overfitting to noise, making it less reliable for direct evaluation.

Recommended Metric Set

Based on the above, the final metric set for ranking wavelets should include:
	•	wavelet_mse
	•	wavelet_psnr
	•	wavelet_energy_entropy
	•	wavelet_sparsity
	•	emd_value
	•	kl_divergence
	•	smoothness
	•	correlation

## Overview Of Scripts

- *`generate_token_frequency_wavelet_analysis.py`*: This script generates a 


- **`generate_wavelet_signal_processing.py`**: This script provides tools for analyzing token frequency as a signal using wavelet and signal processing methods. Below is a detailed breakdown of its functionality:

| Function                                 | Description                                                                                                                                                         | Parameters                                                                                                                                                                                                                     | Returns                                                                                                           |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `process_wavelet_results`               | Processes and cleans wavelet results, handling NaN, infinite values, and skipped results.                                                                           | `results` (list): Wavelet analysis results. <br> `skipped_results` (list): Skipped wavelet results. <br> `signal_type` (str): Signal type. <br> `metrics` (list): Metrics to validate (default: `['wavelet_energy_entropy', 'wavelet_sparsity']`). | Tuple: Cleaned results (`pd.DataFrame`), combined skipped results (`pd.DataFrame`).                              |
| `energy_entropy_ratio`                  | Computes energy-to-entropy ratio for wavelet coefficients.                                                                                                          | `coeffs` (list): Wavelet decomposition coefficients.                                                                                                                                                                         | `float`: Energy-to-entropy ratio.                                                                                 |
| `adaptive_threshold`                    | Calculates an adaptive threshold for sparsity using the Median Absolute Deviation (MAD).                                                                            | `coeffs` (list): Wavelet decomposition coefficients.                                                                                                                                                                         | `float`: Adaptive threshold.                                                                                      |
| `adaptive_sparsity_measure`             | Measures sparsity of wavelet coefficients using an adaptive threshold.                                                                                              | `coeffs` (list): Wavelet decomposition coefficients.                                                                                                                                                                         | Tuple: Sparsity (`float`), adaptive threshold (`float`).                                                          |
| `wavelet_entropy`                       | Calculates wavelet entropy as a measure of signal complexity.                                                                                                       | `coeffs` (list): Wavelet decomposition coefficients.                                                                                                                                                                         | `float`: Wavelet entropy.                                                                                        |
| `signal_smoothness`                     | Computes signal smoothness based on second-order differences.                                                                                                       | `signal` (np.ndarray): Signal to analyze.                                                                                                                                                                                    | `float`: Signal smoothness.                                                                                       |
| `correlation_coefficients`              | Calculates the correlation coefficient between original and reconstructed signals.                                                                                   | `original` (np.ndarray): Original signal. <br> `reconstructed` (np.ndarray): Reconstructed signal.                                                                                                                           | `float`: Correlation coefficient.                                                                                 |
| `signal_variance_across_levels`         | Calculates variance of wavelet coefficients across decomposition levels.                                                                                            | `coeffs` (list): Wavelet decomposition coefficients.                                                                                                                                                                         | `list`: Variance values across levels.                                                                            |
| `compute_additional_wavelet_features`   | Computes additional features from wavelet coefficients and reconstructed signals, such as entropy, smoothness, and correlation.                                      | `coeffs` (list): Wavelet decomposition coefficients. <br> `reconstructed_signal` (np.ndarray): Reconstructed signal. <br> `original_signal` (np.ndarray): Original signal.                                                   | `dict`: Dictionary of additional computed features.                                                               |
| `calculate_signal_metrics`              | Computes metrics for a given signal, including FFT-based features, peaks, autocorrelation, and spectral features.                                                   | `tokens_signal` (np.ndarray): Signal to analyze. <br> `use_signal_type` (str): Signal type. <br> `min_tokens` (float): Minimum observed tokens per page. <br> Additional optional parameters for peak detection and debugging. | `dict`: Calculated signal metrics.                                                                                |
| `pad_signal`                            | Pads a signal to meet requirements for Stationary Wavelet Transform (SWT).                                                                                          | `signal` (np.ndarray): Input signal. <br> `max_level` (int): Maximum decomposition level.                                                                                                                                   | Tuple: Padded signal (`np.ndarray`), padding status (`bool`).                                                     |
| `validate_max_level`                    | Validates and adjusts the maximum decomposition level for SWT based on signal length.                                                                                | `signal_length` (int): Length of the signal. <br> `requested_max_level` (int): Requested maximum level.                                                                                                                      | Tuple: Validated level (`int`), original level status (`bool`).                                                   |
| `is_wavelet_compatible`                 | Checks if a wavelet is compatible with the signal length at the specified level.                                                                                     | `signal_length` (int): Length of the signal. <br> `wavelet` (str): Wavelet name. <br> `level` (int): Decomposition level.                                                                                                    | `bool`: Compatibility status.                                                                                     |
| `determine_scales`                      | Dynamically determines wavelet scales for Continuous Wavelet Transform (CWT).                                                                                       | `signal_length` (int): Signal length. <br> `max_scale` (int): Maximum scale (default: 128). <br> `dynamic` (bool): Whether to calculate scales dynamically (default: `True`).                                                | `np.ndarray`: Array of wavelet scales.                                                                            |
| `process_dwt_wavelet`                   | Processes a single wavelet for Discrete Wavelet Transform (DWT).                                                                                                    | `signal` (np.ndarray): Signal to analyze. <br> `wavelet` (str): Wavelet name. <br> `modes` (list): Extension modes. <br> `signal_type` (str): Signal type.                                                                   | Tuple: Results (`list`), skipped wavelets (`list`).                                                               |
| `evaluate_dwt_performance`              | Evaluates DWT performance for a given signal, testing multiple wavelets and modes.                                                                                  | `signal` (np.ndarray): Signal to analyze. <br> `wavelets` (list): List of wavelet names. <br> `modes` (list): Signal extension modes. <br> `signal_type` (str): Signal type.                                                 | Tuple: Results (`pd.DataFrame`), skipped wavelets (`pd.DataFrame`).                                               |
| `evaluate_dwt_performance_parallel`     | Parallelized evaluation of DWT performance across multiple wavelets.                                                                                                | Same as `evaluate_dwt_performance`.                                                                                                                                                                                           | Same as `evaluate_dwt_performance`.                                                                               |
| `process_cwt_wavelet`                   | Processes a single wavelet for Continuous Wavelet Transform (CWT).                                                                                                  | `signal` (np.ndarray): Signal to analyze. <br> `wavelet` (str): Wavelet name. <br> `scales` (np.ndarray): Wavelet scales. <br> `signal_type` (str): Signal type.                                                             | Tuple: Results (`list`), skipped wavelets (`list`).                                                               |
| `evaluate_cwt_performance`              | Evaluates CWT performance for a given signal, testing multiple wavelets and scales.                                                                                 | `signal` (np.ndarray): Signal to analyze. <br> `wavelets` (list): List of wavelet names. <br> `signal_type` (str): Signal type. <br> `max_scale` (int): Maximum scale. <br> `dynamic_scales` (bool): Use dynamic scales.       | Tuple: Results (`pd.DataFrame`), skipped wavelets (`pd.DataFrame`).                                               |
| `evaluate_cwt_performance_parallel`     | Parallelized evaluation of CWT performance across multiple wavelets.                                                                                                | Same as `evaluate_cwt_performance`.                                                                                                                                                                                            | Same as `evaluate_cwt_performance`.                                                                               |
| `process_swt_wavelet`                   | Processes a single wavelet for Stationary Wavelet Transform (SWT).                                                                                                  | `signal` (np.ndarray): Signal to analyze. <br> `wavelet` (str): Wavelet name. <br> `signal_type` (str): Signal type. <br> `max_level` (int): Maximum decomposition level.                                                     | Tuple: Results (`list`), skipped wavelets (`list`).                                                               |
| `evaluate_swt_performance`              | Evaluates SWT performance for a given signal, testing multiple wavelets.                                                                                            | `signal` (np.ndarray): Signal to analyze. <br> `wavelets` (list): List of wavelet names. <br> `signal_type` (str): Signal type. <br> `max_level` (int): Maximum decomposition level.                                          | Tuple: Results (`pd.DataFrame`), skipped wavelets (`pd.DataFrame`).                                               |
| `evaluate_swt_performance_parallel`     | Parallelized evaluation of SWT performance across multiple wavelets.                                                                                                | Same as `evaluate_swt_performance`.                                                                                                                                                                                            | Same as `evaluate_swt_performance`.                                                                               |