# Scripts Folder

*Currently a work-in-progress*

This folder contains a variety of scripts for periodical issue segmentation and modeling:

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


## Serial Token Frequency As Wavelet & Signal Processing Analysis Scripts Notes

This folder contains the code and documentation for analyzing token frequency data extracted from OCR text as a signal, enabling the use of signal processing techniques to extract meaningful patterns. Essentially, it treats token frequency as a one dimensional waveform, allowing us to apply signal processing techniques to extract meaningful patterns and trends.

### Core Concepts & Assumptions

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

