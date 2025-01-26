# Serial Token Frequency As Wavelet & Signal Processing Analysis Scripts Notes

This folder contains the code and documentation for analyzing token frequency data extracted from OCR text as a signal, enabling the use of signal processing techniques to extract meaningful patterns. Essentially, it treats token frequency as a one dimensional waveform, allowing us to apply signal processing techniques to extract meaningful patterns and trends.

## Core Concepts & Assumptions

Metrics to Include

These metrics directly evaluate the quality of the wavelet representation or its alignment with the original signal:

1. wavelet_mse (Mean Squared Error):

   - Why Include: A lower MSE indicates better reconstruction fidelity. It’s a core measure of how well the wavelet transform approximates the original signal.

2. wavelet_psnr (Peak Signal-to-Noise Ratio):

   - Why Include: Complements MSE by quantifying reconstruction quality on a logarithmic scale. It’s particularly useful when MSE alone doesn’t fully capture perceptual quality.

3. wavelet_energy_entropy:

   - Why Include: Indicates the balance between energy distribution and entropy, highlighting how well the wavelet preserves signal structure. Particularly relevant for token frequency signals, which may have structural patterns.

4. wavelet_sparsity:

   - Why Include: Reflects the compactness of the wavelet representation, which can be valuable for identifying efficient wavelet transforms.

5. emd_value (Earth Mover’s Distance):

   - Why Include: Measures distributional differences between the original and reconstructed signals, providing insight into alignment beyond pixel-wise errors.

6. kl_divergence:

   - Why Include: Captures information-theoretic differences between the original and reconstructed signals. Useful for evaluating how well the transform preserves statistical characteristics.

7. smoothness:

    - Why Include: Highlights the degree to which the reconstructed signal avoids oscillatory noise. This can be valuable if smoothness is desired in your analysis.

8. correlation:

   - Why Include: Measures the linear relationship between the original and reconstructed signals, which is crucial for preserving signal integrity.

Metrics to Exclude

These metrics are better treated as informational rather than included in rankings:
	1.	wavelet_adaptive_threshold:
	•	Why Exclude: It’s primarily descriptive of the sparsity threshold derived for the wavelet coefficients. While informative, it doesn’t directly evaluate performance.
	2.	signal_length:
	•	Why Exclude: This is an intrinsic property of the signal and doesn’t contribute to evaluating wavelet performance.
	3.	decomposition_levels / scales_used:
	•	Why Exclude: While these can influence performance, they are better used to describe configurations rather than directly ranking wavelets.

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