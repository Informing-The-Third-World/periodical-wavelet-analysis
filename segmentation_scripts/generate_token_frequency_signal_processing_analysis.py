# Standard library imports
import os
import sys
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from tqdm import tqdm
from rich.console import Console
import pywt
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

# Local application imports
sys.path.append("..")
from segmentation_scripts.utils import read_csv_file, get_data_directory_path, save_chart, process_file, generate_table

# Disable max rows for Altair
alt.data_transformers.disable_max_rows()

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

def energy_entropy_ratio(coeffs: list) -> float:
	"""
	Compute energy-to-entropy ratio for wavelet coefficients.

	Parameters:
	-----------
	coeffs : list of np.ndarray
		Wavelet decomposition coefficients.

	Returns:
	--------
	ratio : float
		Energy-to-entropy ratio.
	"""
	# Use absolute values of coefficients for energy and entropy computation
	magnitudes = np.abs(coeffs)
	total_energy = np.sum(magnitudes ** 2)
	entropy = -np.sum(
		(magnitudes ** 2 / total_energy) * np.log2(magnitudes ** 2 / total_energy + 1e-12)
	)  # Add a small constant to avoid log(0)
	return total_energy / (entropy if entropy > 0 else 1e-12)  # Avoid division by zero

def sparsity_measure(coeffs: list, threshold: float = 1e-2) -> float:
	"""
	Measure sparsity of wavelet coefficients.

	Parameters:
	-----------
	coeffs : list of np.ndarray
		Wavelet decomposition coefficients.
	threshold : float, optional
		Threshold for near-zero coefficients.

	Returns:
	--------
	sparsity : float
		Percentage of near-zero coefficients.
	"""
	total_coeffs = np.abs(np.concatenate(coeffs))
	
	# Normalize coefficients to scale values between 0 and 1
	normalized_coeffs = total_coeffs / (np.max(total_coeffs) if np.max(total_coeffs) > 0 else 1)
	
	# Compute sparsity as percentage of near-zero coefficients
	sparsity = np.sum(normalized_coeffs < threshold) / len(normalized_coeffs)
	return sparsity

def evaluate_dwt_performance(signal: np.ndarray, wavelets: list, modes: list, signal_type: str) -> tuple:
	"""
	Evaluate DWT wavelet decomposition using MSE, energy-to-entropy ratio, and sparsity for a given signal type.

	Parameters:
	-----------
	signal : np.ndarray
		Signal to analyze (raw or smoothed tokens).
	wavelets : list of str
		List of wavelet names to test.
	modes : list of str
		List of signal extension modes to test.
	signal_type : str
		The type of signal being analyzed ('raw' or 'smoothed').

	Returns:
	--------
	tuple:
		- total_results: pd.DataFrame
		  DataFrame containing wavelet analysis results.
		- skipped_wavelets_df: pd.DataFrame
		  DataFrame containing skipped wavelets.
	"""
	results = []
	skipped_wavelets = []
	for wavelet in tqdm(wavelets, desc=f"Testing Wavelets for {signal_type}"):
		try:
			wavelet_filter_len = pywt.Wavelet(wavelet).dec_len
			if len(signal) < wavelet_filter_len:
				raise ValueError(f"Signal is too short for wavelet {wavelet}")

			max_level = pywt.dwt_max_level(len(signal), filter_len=wavelet_filter_len)
			for level in range(1, max_level + 1):
				for mode in modes:
					try:
						# Decompose signal
						coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)
						# Check if all coefficient lengths are consistent
						coeff_shapes = [len(c) for c in coeffs]
						if len(set(coeff_shapes)) > 1:
							skipped_wavelets.append({'wavelet': wavelet, 'level': level, 'mode': mode, 'coeff_shapes': coeff_shapes, 'error': "Inconsistent coefficient lengths", 'signal_length': len(signal), 'signal_type': signal_type})
							continue
						# Reconstruct signal
						reconstructed_signal = pywt.waverec(coeffs, wavelet, mode=mode)[:len(signal)]

						# Compute Metrics
						mse = np.mean((signal - reconstructed_signal) ** 2)
						energy_entropy = energy_entropy_ratio(coeffs)
						sparsity = sparsity_measure(coeffs)

						# Append Results
						results.append({
							'signal_type': signal_type,
							'wavelet': wavelet,
							'wavelet_level': level,
							'wavelet_mode': mode,
							'wavelet_mse': mse,
							'wavelet_energy_entropy': energy_entropy,
							'wavelet_sparsity': sparsity,
							'signal_length': len(signal)
						})
					except ValueError as e:
						skipped_wavelets.append({'wavelet': wavelet, 'level': level, 'mode': mode, 'error': str(e), 'signal_length': len(signal), 'signal_type': signal_type})
		except Exception as e:
			skipped_wavelets.append({'wavelet': wavelet, 'level': None, 'mode': None, 'error': str(e), 'signal_length': len(signal), 'signal_type': signal_type})

	skipped_wavelets_df = pd.DataFrame(skipped_wavelets)
	console.print(f"Skipped wavelets: {len(skipped_wavelets_df)}", style="bright_red")
	total_results = pd.DataFrame(results)
	console.print(f"Total DWT results: {len(total_results)}", style="bright_green")
	return total_results, skipped_wavelets_df

def determine_scales(signal_length: int, max_scale: int = 128, dynamic: bool = True) -> np.ndarray:
    """
    Determine wavelet scales dynamically based on signal length.

    Parameters:
    -----------
    signal_length : int
        Length of the signal to analyze.
    max_scale : int
        Maximum scale to consider.
    dynamic : bool
        Whether to calculate scales dynamically based on the signal length.

    Returns:
    --------
    scales : np.ndarray
        Array of scales for the wavelet transform.
    """
    if dynamic:
        # Use half the signal length or cap at max_scale
        scales = np.arange(1, min(signal_length // 2, max_scale))
    else:
        # Use a predefined range of scales
        scales = np.arange(1, max_scale)
    return scales

def evaluate_cwt_performance(signal: np.ndarray, wavelets: list, signal_type: str, max_scale: int = 128, dynamic_scales: bool = True) -> tuple:
    """
    Evaluate Continuous Wavelet Transform (CWT) using MSE, energy-to-entropy ratio, and sparsity for a given signal type.

    Parameters:
    -----------
    signal : np.ndarray
        Signal to analyze (raw or smoothed tokens).
    wavelets : list of str
        List of wavelet names to test.
    signal_type : str
        The type of signal being analyzed ('raw' or 'smoothed').
    max_scale : int
        Maximum scale to consider for the wavelet transform.
    dynamic_scales : bool
        Whether to determine scales dynamically based on the signal length.

    Returns:
    --------
    total_results : pd.DataFrame
        DataFrame containing wavelet analysis results.
    skipped_results : pd.DataFrame
        DataFrame containing skipped wavelets.
    """
    # Dynamically determine scales
    scales = determine_scales(len(signal), max_scale=max_scale, dynamic=dynamic_scales)
    
    results = []
    skipped_results = []
    
    for wavelet in tqdm(wavelets, desc=f"Testing CWT Wavelets for {signal_type}"):
        try:
            # Perform Continuous Wavelet Transform
            coeffs, _ = pywt.cwt(signal, scales=scales, wavelet=wavelet)
            
            # Compute Metrics
            total_energy = np.sum(coeffs ** 2)
            entropy = -np.sum(coeffs ** 2 / total_energy * np.log2(coeffs ** 2 / total_energy), axis=None)
            energy_entropy = total_energy / entropy if entropy > 0 else np.inf
            sparsity = np.sum(np.abs(coeffs) < 1e-3) / coeffs.size

            # Append Results
            results.append({
                'signal_type': signal_type,
                'wavelet': wavelet,
                'wavelet_energy_entropy': energy_entropy,
                'wavelet_sparsity': sparsity,
                'signal_length': len(signal),
                'scales_used': len(scales)  # Number of scales used for reference
            })
        except Exception as e:
            skipped_results.append({'wavelet': wavelet, 'error': str(e), 'signal_length': len(signal), 'signal_type': signal_type, 'scales_used': len(scales)})
    
    total_results = pd.DataFrame(results)
    
    metrics = ['wavelet_energy_entropy', 'wavelet_sparsity']
    subset_results = total_results.replace([np.inf, -np.inf], np.nan).dropna(subset=metrics).reset_index(drop=True)
    console.print(f"Total CWT results: {len(subset_results)}", style="bright_green")
    
    error_results = total_results[~total_results.index.isin(subset_results.index)]
    skipped_results_df = pd.DataFrame(skipped_results)
    combined_skipped_results = pd.concat([error_results, skipped_results_df], ignore_index=True)
    console.print(f"Skipped CWT results: {len(combined_skipped_results)}", style="bright_red")
    
    return total_results, combined_skipped_results

def evaluate_swt_performance(signal: np.ndarray, wavelets: list, signal_type: str, max_level: int = 5) -> tuple:
    """
    Evaluate Stationary Wavelet Transform (SWT) using energy-to-entropy ratio and sparsity for a given signal type.

    Parameters:
    -----------
    signal : np.ndarray
        Signal to analyze (raw or smoothed tokens).
    wavelets : list of str
        List of wavelet names to test.
    signal_type : str
        The type of signal being analyzed ('raw' or 'smoothed').
    max_level : int
        Maximum decomposition level for SWT.
    
    Returns:
    --------
    total_results : pd.DataFrame
        DataFrame containing wavelet analysis results.
    skipped_results : pd.DataFrame
        DataFrame containing skipped wavelets.
    """
    results = []
    skipped_wavelets = []

    for wavelet in tqdm(wavelets, desc=f"Testing SWT Wavelets for {signal_type}"):
        try:
            # Decompose signal using SWT
            coeffs = pywt.swt(signal, wavelet=wavelet, level=max_level)
            approx_coeffs, detail_coeffs = zip(*coeffs)  # Separate approximation and detail coefficients
            
            # Compute metrics
            total_energy = np.sum([np.sum(np.array(c) ** 2) for c in detail_coeffs])
            entropy = -np.sum(
                [np.sum(np.array(c) ** 2 / total_energy * np.log2(np.array(c) ** 2 / total_energy + 1e-12)) 
                 for c in detail_coeffs]
            )
            energy_entropy = total_energy / (entropy if entropy > 0 else np.inf)
            sparsity = np.sum([np.sum(np.abs(c) < 1e-3) for c in detail_coeffs]) / np.sum([c.size for c in detail_coeffs])

            # Append results
            results.append({
                'signal_type': signal_type,
                'wavelet': wavelet,
                'wavelet_energy_entropy': energy_entropy,
                'wavelet_sparsity': sparsity,
                'signal_length': len(signal),
                'decomposition_levels': max_level
            })
        except Exception as e:
            skipped_wavelets.append({'wavelet': wavelet, 'error': str(e), 'signal_length': len(signal), 'signal_type': signal_type})

    total_results = pd.DataFrame(results)
    skipped_results = pd.DataFrame(skipped_wavelets)
    console.print(f"Total SWT results: {len(total_results)}", style="bright_green")
    console.print(f"Skipped SWT results: {len(skipped_results)}", style="bright_red")
    return total_results, skipped_results

def determine_best_wavelet_representation(results_df: pd.DataFrame, signal_type: str, weights: dict = None, is_combined: bool = False) -> tuple:
	"""
	Determine the best wavelet representation based on MSE, Energy-to-Entropy Ratio, and Sparsity within a volume. Normalize and combine scores to rank the results for both normalized and z-scored metrics across DWT and CWT.

	Parameters:
	-----------
	results_df : pd.DataFrame
		DataFrame containing Wavelet metrics.
	signal_type : str
		Type of signal being analyzed (e.g., "DWT" or "CWT").
	weights : dict, optional
		Dictionary of weights for each metric. Example:
		{'MSE': 0.5, 'Energy-to-Entropy': 0.3, 'Sparsity': 0.2}
	is_combined : bool, optional
		Flag indicating whether the results are combined. If True, the prefix 'combined_' is used.

	Returns:
	--------
	best_config : pd.Series
		The row containing the best wavelet, level, and mode.
	ranked_results : pd.DataFrame
		DataFrame with combined scores and rankings.
	correlation_norm_zscore : float
		Correlation between normalized and zscored combined scores. Useful for checking consistency and identifying outliers.
	"""
	console.print(f"Results for {signal_type} Wavelet Analysis")
	# Handle zero variance in sparsity
	if 'wavelet_sparsity' in results_df.columns and results_df['wavelet_sparsity'].nunique() == 1:
		console.print(f"[yellow]Dropping 'wavelet_sparsity' due to zero variance.[/yellow]")
		results_df = results_df.drop(columns=['wavelet_sparsity'])
		weights.pop('wavelet_sparsity', None)
	# Handle extreme values in energy_entropy
	if 'wavelet_energy_entropy' in results_df.columns:
		results_df['wavelet_energy_entropy'] = np.log1p(np.abs(results_df['wavelet_energy_entropy']))
	# Handle complex-valued metrics only
	for column in ['wavelet_energy_entropy']:
		if column in results_df.columns and np.iscomplexobj(results_df[column].values):
			console.print(f"Converting {column} to magnitudes due to complex values.", style="yellow")
			results_df[column] = np.abs(results_df[column])

	# Normalize metrics
	scaler = MinMaxScaler()
	normalized_df = results_df.copy()
	prefix = 'combined_' if is_combined else ''

	# Identify metrics for normalization
	metrics = [col for col in ['wavelet_mse', 'wavelet_energy_entropy', 'wavelet_sparsity'] if col in results_df.columns]
	

	# Normalize specified metrics
	normalized_metrics = [f"{prefix}{metric}_norm" for metric in metrics]
	try:
		normalized_df[normalized_metrics] = scaler.fit_transform(results_df[metrics])
	except ValueError as e:
		console.print(f"[bright_red]Error normalizing metrics: {e}[/bright_red]")
		results_df = results_df.replace([np.inf, -np.inf], np.nan).dropna(subset=metrics).reset_index(drop=True)
		normalized_df = results_df.copy()
		metrics = [col for col in ['wavelet_mse', 'wavelet_energy_entropy', 'wavelet_sparsity'] if col in results_df.columns]
		normalized_metrics = [f"{prefix}{metric}_norm" for metric in metrics]
		normalized_df[normalized_metrics] = scaler.fit_transform(results_df[metrics])
		
	if 'wavelet_mse' in results_df.columns:
		normalized_df[f'{prefix}wavelet_mse_norm'] = 1 - normalized_df[f'{prefix}wavelet_mse_norm']  # Lower MSE is better

	# Compute z-score normalization
	zscore_metrics = [f"{prefix}{metric}_zscore" for metric in metrics]
	normalized_df[zscore_metrics] = (
		(normalized_df[metrics] - normalized_df[metrics].mean()) /
		(normalized_df[metrics].std().replace(0, np.nan))  # Avoid division by zero
	)

	# Compute combined scores
	normalized_df[f'{prefix}wavelet_norm_combined_score'] = sum(
		weights.get(metric, 0) * normalized_df.get(f'{prefix}{metric}_norm', 0) 
		for metric in metrics
	)
	normalized_df[f'{prefix}wavelet_zscore_combined_score'] = sum(
		weights.get(metric, 0) * normalized_df.get(f'{prefix}{metric}_zscore', 0) 
		for metric in metrics
	)

	# Rank results
	ranked_results = normalized_df.sort_values(
		by=[f'{prefix}wavelet_norm_combined_score', f'{prefix}wavelet_zscore_combined_score'], 
		ascending=False
	).reset_index(drop=True)
	ranked_results[f'{prefix}wavelet_rank'] = ranked_results.index + 1

	# Calculate correlation between normalized and z-score combined scores
	correlation_norm_zscore = ranked_results[
		[f'{prefix}wavelet_norm_combined_score', f'{prefix}wavelet_zscore_combined_score']
	].corr().iloc[0, 1]

	# Return the best configuration, ranked results, and correlation
	best_config = ranked_results[0:1]
	return best_config, ranked_results, correlation_norm_zscore

def compare_and_rank_wavelet_metrics(raw_signal: np.ndarray, smoothed_signal: np.ndarray) -> tuple:
	"""
	Compare wavelet metrics for raw and smoothed tokens and determine the best representation.

	Parameters:
	-----------
	raw_signal : np.ndarray
		Raw tokens per page.
	smoothed_signal : np.ndarray
		Smoothed tokens per page.
	wavelets : list of str
		List of wavelet names to test.
	modes : list of str
		List of signal extension modes to test.
	weights : dict
		Weights for MSE, Energy-to-Entropy, and Sparsity (e.g., {'wavelet_mse': 0.5, 'wavelet_energy_entropy': 0.3, 'wavelet_sparsity': 0.2}).

	Returns:
	--------
	tuple:
		- ranked_combined_results: pd.DataFrame
		  DataFrame containing combined wavelet analysis results.
		- best_combined_results: pd.Series
		  Series containing the best combined wavelet configuration.
		- combined_correlation_score: float
		  Correlation between normalized and zscored combined scores.
		- dwt_combined_skipped_results: pd.DataFrame
		  DataFrame containing skipped DWT wavelets.
		- cwt_combined_skipped_results: pd.DataFrame
		  DataFrame containing skipped CWT wavelets.
	"""
	dwt_wavelets = pywt.wavelist(kind='discrete')
	cwt_wavelets = pywt.wavelist(kind='continuous')
	modes = pywt.Modes.modes
	weights = {'wavelet_mse': 0.5, 'wavelet_energy_entropy': 0.3, 'wavelet_sparsity': 0.2}  # Weights for metrics
	# Evaluate metrics for DWT
	dwt_raw_results, dwt_raw_skipped_results = evaluate_dwt_performance(raw_signal, dwt_wavelets, modes, 'raw')
	dwt_smoothed_results, dwt_smoothed_skipped_results = evaluate_dwt_performance(smoothed_signal, dwt_wavelets, modes, 'smoothed')

	# Evaluate metrics for CWT
	cwt_raw_results, cwt_raw_skipped_results = evaluate_cwt_performance(raw_signal, cwt_wavelets, 'raw')
	cwt_smoothed_results, cwt_smoothed_skipped_results = evaluate_cwt_performance(smoothed_signal, cwt_wavelets, 'smoothed')

	# Evaluate metrics for SWT
	swt_raw_results, swt_raw_skipped_results = evaluate_swt_performance(raw_signal, dwt_wavelets, 'raw')
	swt_smoothed_results, swt_smoothed_skipped_results = evaluate_swt_performance(smoothed_signal, dwt_wavelets, 'smoothed')

	# Combine results
	dwt_combined_results = pd.concat([dwt_raw_results, dwt_smoothed_results], ignore_index=True)
	dwt_combined_skipped_results = pd.concat([dwt_raw_skipped_results, dwt_smoothed_skipped_results], ignore_index=True)
	cwt_combined_results = pd.concat([cwt_raw_results, cwt_smoothed_results], ignore_index=True)
	cwt_combined_skipped_results = pd.concat([cwt_raw_skipped_results, cwt_smoothed_skipped_results], ignore_index=True)
	swt_combined_results = pd.concat([swt_raw_results, swt_smoothed_results], ignore_index=True)
	swt_combined_skipped_results = pd.concat([swt_raw_skipped_results, swt_smoothed_skipped_results], ignore_index=True)

	# Determine best representations
	best_dwt, ranked_dwt, dwt_correlation_score = determine_best_wavelet_representation(dwt_combined_results, "DWT", weights, False)
	best_cwt, ranked_cwt, cwt_correlation_score = determine_best_wavelet_representation(cwt_combined_results, "CWT", weights, False)
	best_swt, ranked_swt, swt_correlation_score = determine_best_wavelet_representation(swt_combined_results, "SWT", weights, False)

	ranked_dwt['wavelet_type'] = 'DWT'
	ranked_cwt['wavelet_type'] = 'CWT'
	ranked_swt['wavelet_type'] = 'SWT'
	# Combine DWT and CWT results
	combined_results = pd.concat([ranked_dwt, ranked_cwt, ranked_swt], ignore_index=True)

	# Determine overall best representation
	best_combined_results, ranked_combined_results, combined_correlation_score = determine_best_wavelet_representation(
		combined_results, "Combined", weights, True
	)

	# Display results
	if len(best_dwt) > 0:
		generate_table(best_dwt, f"Best DWT Wavelet Configuration (Correlation: {dwt_correlation_score:.2f})")
	if len(best_cwt) > 0:
		generate_table(best_cwt, f"Best CWT Wavelet Configuration (Correlation: {cwt_correlation_score:.2f})")
	if len(best_swt) > 0:
		generate_table(best_swt, f"Best SWT Wavelet Configuration (Correlation: {swt_correlation_score:.2f})")
	if len(best_combined_results) > 0:
		generate_table(best_combined_results, f"Best Combined Wavelet Configuration (Correlation: {combined_correlation_score:.2f})")

	return ranked_combined_results, best_combined_results, combined_correlation_score, dwt_combined_skipped_results, cwt_combined_skipped_results, swt_combined_skipped_results

def calculate_signal_metrics(
	tokens_signal: np.ndarray,
	use_signal_type: str,
	min_tokens: float,
	prominence=1.0,
	distance=5,
	verbose=False
) -> dict:
	"""
	Calculate metrics for a given signal, including dominant frequency, dynamic cutoff,
	relative peak detection, autocorrelation, signal envelope, and spectral features.

	Parameters
	----------
	tokens_signal : np.ndarray
		The signal to be analyzed.
	use_signal_type : str
		The type of page being analyzed (e.g., raw or smoothed "tokens_per_page").
	min_tokens : float
		The minimum observed tokens per page in the original scale.
	prominence : float, optional
		Minimum prominence of peaks for relative detection.
	distance : int, optional
		Minimum distance between peaks for relative detection.
	verbose : bool, optional
		Whether to log detailed metrics for debugging.

	Returns
	-------
	dict
		Results containing signal metrics.
	"""
	try:
		# Perform FFT for frequency analysis
		tokens_fft = fft(tokens_signal)
		frequencies = np.fft.fftfreq(len(tokens_fft))

		# Analyze positive frequencies and amplitudes
		positive_frequencies = frequencies[:len(frequencies)//2]
		positive_amplitudes = np.abs(tokens_fft[:len(tokens_fft)//2])

		# Find FFT peaks
		peaks, _ = find_peaks(positive_amplitudes[1:])
		num_peaks = len(peaks)
		peak_amplitude = np.max(positive_amplitudes[1:][peaks]) if num_peaks > 0 else None
		dominant_frequency = (
			positive_frequencies[peaks[np.argmax(positive_amplitudes[1:][peaks])]] 
			if num_peaks > 0 else None
		)

		# Calculate dynamic cutoff
		dynamic_cutoff_signal = (
			max(np.median(tokens_signal) - (peak_amplitude or 0), np.percentile(tokens_signal, 10))
			if num_peaks > 0 else np.percentile(tokens_signal, 10)
		)
		dynamic_cutoff_original_scale = max(dynamic_cutoff_signal, min_tokens)

		# Perform relative peak detection
		prominence = prominence if prominence is not None else np.std(tokens_signal) * 0.1
		distance = distance if distance is not None else max(1, len(tokens_signal) // 20)
		relative_peaks, relative_properties = find_peaks(tokens_signal, prominence=prominence, distance=distance)
		relative_num_peaks = len(relative_peaks)
		avg_prominence = np.mean(relative_properties["prominences"]) if relative_num_peaks > 0 else None

		# Calculate autocorrelation
		autocorr = np.correlate(tokens_signal, tokens_signal, mode='full')
		max_autocorr = np.max(autocorr[len(autocorr)//2:])

		# Signal envelope
		upper_envelope = np.max(np.abs(tokens_signal))
		lower_envelope = -upper_envelope

		# Spectral features
		spectral_magnitude = np.sum(positive_amplitudes)
		spectral_centroid = (
			np.sum(positive_frequencies * positive_amplitudes) / spectral_magnitude 
			if spectral_magnitude > 0 else None
		)
		spectral_bandwidth = (
			np.sqrt(np.sum((positive_frequencies - spectral_centroid) ** 2 * positive_amplitudes) / spectral_magnitude) 
			if spectral_centroid else None
		)

		# Logging (controlled by `verbose`)
		if verbose:
			console.print(f"[bright_cyan]Metrics for {use_signal_type}[/bright_cyan]")
			console.print(f"Dominant Frequency: {dominant_frequency}")
			console.print(f"Dynamic Cutoff: {dynamic_cutoff_original_scale}")
			console.print(f"Number of Peaks: {num_peaks}")
			console.print(f"Peak Amplitude: {peak_amplitude}")
			console.print(f"Relative Peaks: {relative_num_peaks}")
			console.print(f"Average Prominence: {avg_prominence}")
			console.print(f"Max Autocorrelation: {max_autocorr}")
			console.print(f"Spectral Centroid: {spectral_centroid}")
			console.print(f"Spectral Bandwidth: {spectral_bandwidth}")
			console.print(f"Upper Envelope: {upper_envelope}")
			console.print(f"Lower Envelope: {lower_envelope}")
			console.print(f"Spectral Magnitude: {spectral_magnitude}")

		return {
			"signal_type": use_signal_type,
			"dominant_frequency": dominant_frequency,
			"dynamic_cutoff": dynamic_cutoff_original_scale,
			"num_peaks": num_peaks,
			"peak_amplitude": peak_amplitude,
			"relative_num_peaks": relative_num_peaks,
			"avg_prominence": avg_prominence,
			"max_autocorrelation": max_autocorr,
			"upper_envelope": upper_envelope,
			"lower_envelope": lower_envelope,
			"spectral_centroid": spectral_centroid,
			"spectral_bandwidth": spectral_bandwidth,
			"spectral_magnitude": spectral_magnitude,
			"positive_frequencies": positive_frequencies,
			"positive_amplitudes": positive_amplitudes,
		}
	except Exception as e:
		console.print(f"[bright_red]Error calculating metrics for {use_signal_type}: {e}[/bright_red]")
		return {}

def process_tokens(file_path: str, preidentified_periodical: bool, should_filter_greater_than_numbers: bool, should_filter_implied_zeroes: bool) -> tuple:
	"""
	Process tokens from the given file and return the processed DataFrame along with normalized token and digit signals.

	Parameters
	----------
	file_path : str
		The path to the file containing the token data.
	preidentified_periodical : bool
		Flag indicating whether the periodical is pre-identified.
	should_filter_greater_than_numbers : bool
		Flag indicating whether to filter out numbers greater than the max possible page number.
	should_filter_implied_zeroes : bool
		Flag indicating whether to filter out implied zeroes.

	Returns
	-------
	tuple
		A tuple containing the merged expanded DataFrame, the grouped DataFrame, the raw token signal, and the smoothed token signal.
	"""
	expanded_df, subset_digits, grouped_df = process_file(file_path, preidentified_periodical, should_filter_greater_than_numbers, should_filter_implied_zeroes)
	
	# Merge metadata if not already present
	if 'enumeration_chronology' not in expanded_df.columns:
		metadata_file_path = file_path.replace("_individual_tokens.csv", "_metadata.csv")
		metadata_df = read_csv_file(metadata_file_path)
		expanded_df = expanded_df.merge(metadata_df, on=['periodical_name', 'htid', 'record_url'], how='left')
	
	subset_cols = ['page_number', 'tokens_per_page', 'original_page_number', 'htid', 'title', 'pub_date', 'enumeration_chronology', 'type_of_resource', 'title', 'date_created', 'pub_date', 'language', 'access_profile', 'isbn', 'issn', 'lccn', 'oclc', 'page_count', 'feature_schema_version', 'access_rights', 'alternate_title', 'category', 'genre_ld', 'genre', 'contributor_ld', 'contributor', 'handle_url', 'source_institution_ld', 'source_institution', 'lcc', 'type', 'is_part_of', 'last_rights_update_date', 'pub_place_ld', 'pub_place', 'main_entity_of_page', 'publisher_ld','publisher', 'record_url', 'periodical_name'] 
	if preidentified_periodical:
		subset_cols = subset_cols + ['start_issue', 'end_issue', 'type_of_page']
	# Select relevant columns and drop duplicates
	subset_expanded_df = expanded_df[subset_cols].drop_duplicates()
	
	min_subset_digits = subset_digits[['original_page_number', 'digits_per_page', 'page_number']].drop_duplicates()
	
	# Merge the token and digit data
	merged_expanded_df = subset_expanded_df.merge(min_subset_digits, on=['original_page_number', 'page_number'], how='left')
	merged_expanded_df['tokens_per_page'] = merged_expanded_df['tokens_per_page'].fillna(0)
	merged_expanded_df['digits_per_page'] = merged_expanded_df['digits_per_page'].fillna(0)
	merged_expanded_df = merged_expanded_df.sort_values(by='page_number')
	
	# Apply smoothing (moving average)
	merged_expanded_df['smoothed_tokens_per_page'] = (
		merged_expanded_df['tokens_per_page']
		.where(merged_expanded_df['tokens_per_page'] > 0)
		.rolling(window=5, center=True)
		.mean()
		.fillna(0)
	)
	# Standardize smoothed signals
	merged_expanded_df['standardized_tokens_per_page'] = (
		(merged_expanded_df['smoothed_tokens_per_page'] - merged_expanded_df['smoothed_tokens_per_page'].mean()) 
		/ merged_expanded_df['smoothed_tokens_per_page'].std()
	)
	merged_expanded_df['smoothed_digits_per_page'] = (
		merged_expanded_df['digits_per_page']
		.where(merged_expanded_df['digits_per_page'] > 0)
		.rolling(window=5, center=True)
		.mean()
		.fillna(0)
	)

	merged_expanded_df['standardized_digits_per_page'] = (
		(merged_expanded_df['smoothed_digits_per_page'] - merged_expanded_df['smoothed_digits_per_page'].mean()) 
		/ merged_expanded_df['smoothed_digits_per_page'].std()
	)
	
	table_cols = ['page_number', 'tokens_per_page', 'smoothed_tokens_per_page', 'standardized_tokens_per_page', 'digits_per_page', 'smoothed_digits_per_page', 'standardized_digits_per_page'] 
	table_title = "Token and Digit Data"
	generate_table(merged_expanded_df[table_cols].head(2), table_title)
	
	# Normalize signals for FFT and autocorrelation
	tokens_raw_signal = np.nan_to_num(merged_expanded_df['tokens_per_page'].values, nan=0.0, posinf=0.0, neginf=0.0)
	tokens_smoothed_signal = np.nan_to_num(merged_expanded_df['smoothed_tokens_per_page'].values, nan=0.0, posinf=0.0, neginf=0.0)
	(tokens_raw_signal)
	console.print(f"Raw Signal Length: {len(tokens_raw_signal)}", style="bright_green")
	console.print(f"Smoothed Signal Length: {len(tokens_smoothed_signal)}", style="bright_green")
	return merged_expanded_df, grouped_df, tokens_raw_signal, tokens_smoothed_signal

def plot_volume_frequencies_matplotlib(volume_frequencies: list, periodical_name: str, output_dir: str):
	"""
	Plot all volume frequencies on the same graph and save as an image. Hard coded to use raw tokens positive frequencies and amplitudes but can be modified.

	Parameters:
	- volume_frequencies: List of volume frequency data.
	- periodical_name: Name of the periodical for the title.
	- output_dir: Directory to save the plot.
	"""
	plt.figure(figsize=(14, 8))
	for volume in volume_frequencies:
		plt.plot(
			volume['raw_positive_frequencies'], 
			volume['raw_positive_amplitudes'], 
			label=f"Volume {volume['htid']}"
		)
	plt.title(f'Frequency Spectra of All Volumes in {periodical_name}')
	plt.xlabel('Frequency')
	plt.ylabel('Amplitude')
	plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside plot
	plt.tight_layout()
	plt.savefig(f"{output_dir}/amplitude_vs_frequencies/{periodical_name}_volume_frequencies.png", dpi=300)  # Save at high resolution
	plt.close()  # Close the plot to save memory

def plot_tokens_per_page(volume_frequencies: list, output_dir: str, periodical_name: str):
	"""
	Plot tokens per page over pages for all volumes.

	Parameters:
	- volume_frequencies: List of volume frequency data.
	- output_dir: Directory to save the plot.
	- periodical_name: Name of the periodical for the title.
	"""
	plt.figure(figsize=(14, 8))
	
	for volume in volume_frequencies:
		# Extract tokens per page and page numbers
		tokens_per_page = volume['tokens_per_page']
		page_numbers = volume['page_numbers']

		plt.plot(page_numbers, tokens_per_page, label=f"Volume {volume['htid']}")
	
	plt.title(f'Tokens Per Page Across Volumes in {periodical_name}')
	plt.xlabel('Page Number')
	plt.ylabel('Tokens Per Page')
	plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside plot
	plt.tight_layout()
	plt.savefig(f"{output_dir}/tokens_per_page/{periodical_name}_tokens_per_page.png", dpi=300)  # Save at high resolution
	plt.close()

def check_if_actual_issue(row, grouped_df):
	"""
	Check if the given row corresponds to an actual issue.

	Parameters
	----------
	row : pd.Series
		The row of the DataFrame being checked.
	grouped_df : pd.DataFrame
		Grouped DataFrame with issue boundaries.

	Returns
	-------
	bool
		True if the page is part of an actual issue, False otherwise.
	"""
	subset_grouped_df = grouped_df[grouped_df.first_page == row.page_number]
	return len(subset_grouped_df) > 0

def visualize_annotated_periodicals(merged_expanded_df, grouped_df, output_dir, periodical_name, dynamic_cutoff):
	"""
	Visualize tokens per page for annotated periodicals and calculate the lowest threshold.

	Parameters:
	- expanded_df: The expanded DataFrame with tokens per page.
	- grouped_df: Grouped DataFrame with issue boundaries.
	- output_dir: Directory to save the visualization.
	- periodical_name: Name of the periodical.
	"""
	# Create the base Altair chart
	selection = alt.selection_point(fields=['start_issue'], bind='legend')
	base = alt.Chart(merged_expanded_df[['page_number', 'tokens_per_page', 'start_issue', 'htid']]).mark_line(point=True).encode(
		x=alt.X("page_number:Q", scale=alt.Scale(zero=False)),
		y=alt.Y('tokens_per_page:Q', scale=alt.Scale(zero=False)),
		color='start_issue:N',
		opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
		tooltip=['page_number', 'tokens_per_page', 'start_issue', 'htid']
	).add_params(selection).properties(
		width=600,
		height=300,
		title=f'Tokens per Page per Issue - {periodical_name} for Volume {merged_expanded_df.htid.unique()[0]}'
	)

	# Add the dynamic cutoff line
	cutoff_line = alt.Chart(pd.DataFrame({'y': [dynamic_cutoff]})).mark_rule(color='blue').encode(
		y=alt.Y('y:Q', axis=alt.Axis(title=None))
	)

	# Combine the base chart and the cutoff line
	chart = base + cutoff_line
	# Identify pages below the dynamic cutoff
	lowest_tokens_df = merged_expanded_df[merged_expanded_df['tokens_per_page'] <= dynamic_cutoff]

	tqdm.pandas(desc="Checking if actual issue")
	# Add 'actual_issue' column to the DataFrame
	lowest_tokens_df['actual_issue'] = lowest_tokens_df.progress_apply(
		check_if_actual_issue, args=(grouped_df,), axis=1
	)
	generate_table(lowest_tokens_df[lowest_tokens_df.actual_issue == True], "Lowest Tokens per Page")

	# Sort and print missing issues
	missing_issues = grouped_df[
		(~grouped_df.start_issue.isin(
			lowest_tokens_df[
				lowest_tokens_df.actual_issue == True
			].start_issue
		))
	]
	print(f"We are missing the following issues: {missing_issues.start_issue.unique()}")
	return missing_issues.start_issue.unique().tolist(), chart

def generate_volume_embeddings(volume_paths_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
	"""
	Generate embeddings for each volume in the given DataFrame.

	Parameters:
	- volume_paths_df: DataFrame containing volume paths.
	- output_dir: Directory to save the embeddings.

	Returns:
	- volume_frequencies: List of volume frequencies.
	"""
	volume_frequencies = []
	volume_paths_df = volume_paths_df.reset_index(drop=True)
	volume_paths_df = volume_paths_df.sort_values(by=['table_row_index'])
	periodical_name = volume_paths_df['lowercase_periodical_name'].unique()[0]
	altair_charts = []
	for _, volume in volume_paths_df.iterrows():
		merged_expanded_df, grouped_df, tokens_raw_signal, tokens_smoothed_signal = process_tokens(
			volume['file_path'], 
			volume['is_annotated_periodical'], 
			volume['should_filter_greater_than_numbers'], 
			volume['should_filter_implied_zeroes']
		)
		# Calculate wavelet metrics and signal metrics
		wavelet_results_df, best_wavelet_config, combined_wavelet_correlation, dwt_skipped_results, cwt_skipped_results = compare_and_rank_wavelet_metrics(
			tokens_raw_signal, tokens_smoothed_signal
		)
		signal_types = {
			"raw": merged_expanded_df['tokens_per_page'].values,
			"smoothed": merged_expanded_df['smoothed_tokens_per_page'].values,
		}
		# Calculate metrics for each representation
		signal_metrics_results = []
		for signal_type, signal in signal_types.items():
			result = calculate_signal_metrics(
				tokens_signal=signal,
				use_signal_type=signal_type,
				min_tokens=merged_expanded_df['tokens_per_page'].min(),
				prominence=1.0,
				distance=5,
				verbose=True
			)
			signal_metrics_results.append(result)

		# Convert to DataFrame for easier analysis
		signal_metrics_df = pd.DataFrame(signal_metrics_results)

		# Separate raw and smoothed signals
		raw_signals = signal_metrics_df[signal_metrics_df.signal_type == 'raw'].drop(columns=['signal_type'])
		smoothed_signals = signal_metrics_df[signal_metrics_df.signal_type == 'smoothed'].drop(columns=['signal_type'])

		# Rename columns to include the signal_type
		raw_signals.columns = [f"raw_{col}" for col in raw_signals.columns]
		smoothed_signals.columns = [f"smoothed_{col}" for col in smoothed_signals.columns]

		# Concatenate the DataFrames side by side
		merged_signals = pd.concat([raw_signals.reset_index(drop=True), smoothed_signals.reset_index(drop=True)], axis=1)
		
		if volume['is_annotated_periodical'] and len(grouped_df) > 1:
			missing_issues, chart = visualize_annotated_periodicals(merged_expanded_df, grouped_df, output_dir, volume['lowercase_periodical_name'], merged_signals.raw_dynamic_cutoff.values[0])
			altair_charts.append(chart)
		else:
			missing_issues = []
			chart = None

		# Use dynamic cutoffs for tokens and digits
		merged_expanded_df['is_likely_cover_raw'] = (
			(merged_expanded_df['tokens_per_page'] <= merged_signals.raw_dynamic_cutoff.values[0])
		)
		merged_expanded_df['is_likely_cover_smoothed'] = (
			(merged_expanded_df['smoothed_tokens_per_page'] <= merged_signals.smoothed_dynamic_cutoff.values[0])
		)

		# List pages marked as likely covers
		raw_list_of_covers = merged_expanded_df[merged_expanded_df['is_likely_cover_raw']].page_number.unique().tolist()
		smoothed_list_of_covers = merged_expanded_df[merged_expanded_df['is_likely_cover_smoothed']].page_number.unique().tolist()

		# Convert best_wavelet row to dictionary
		best_wavelet_dict = best_wavelet_config.iloc[0].to_dict()
		merged_signals_dict = merged_signals.iloc[0].to_dict()

		# Append frequencies and metadata
		volume_data = {
			'wavelet_correlation': combined_wavelet_correlation,
			'htid': merged_expanded_df['htid'].unique()[0],
			'lowercase_periodical_name': volume['lowercase_periodical_name'],
			'avg_tokens': merged_expanded_df['tokens_per_page'].mean(),
			'avg_digits': merged_expanded_df['digits_per_page'].mean(),
			'raw_likely_covers': raw_list_of_covers,
			'smoothed_likely_covers': smoothed_list_of_covers,
			'total_pages': merged_expanded_df['page_number'].nunique(),
			'total_tokens': merged_expanded_df['tokens_per_page'].sum(),
			'total_digits': merged_expanded_df['digits_per_page'].sum(),
			'table_row_index': volume['table_row_index'],
			'tokens_per_page': merged_expanded_df['tokens_per_page'].values,
			'page_numbers': merged_expanded_df['page_number'].values,
			'digits_per_page': merged_expanded_df['digits_per_page'].values,
			'missing_issues': missing_issues,
			'volume_classification': volume['volume_classification'],
			'title_classification': volume['title_classification'],
		}
		volume_df = pd.DataFrame([volume_data])
		volume_df = volume_df.drop(columns=['tokens_per_page', 'page_numbers', 'digits_per_page'])
		# Merge the best_wavelet_dict with volume_data
		volume_data.update(best_wavelet_dict)
		volume_data.update(merged_signals_dict)
		# Append to the list of volume frequencies
		volume_frequencies.append(volume_data)
		# Merge the full wavelet results with the signal metrics
		merged_signal_analysis_df = pd.merge(wavelet_results_df, signal_metrics_df, on='signal_type', how='left')
		# Drop the positive frequencies and amplitudes
		merged_signal_analysis_df = merged_signal_analysis_df.drop(columns=['positive_frequencies', 'positive_amplitudes'])
		# Concat the full wavelet results to the volume data and store as CSV
		repeated_volume_df = pd.concat([volume_df] * len(merged_signal_analysis_df), ignore_index=True)

		full_wavelet_results = pd.concat([repeated_volume_df, merged_signal_analysis_df], axis=1)

		# Extract the directory path without the CSV file
		directory_path = os.path.dirname(volume['file_path'])

		# Create the new directory path for wavelet_analysis
		wavelet_analysis_dir = os.path.join(directory_path, 'wavelet_analysis')

		# Create the wavelet_analysis directory if it doesn't exist
		os.makedirs(wavelet_analysis_dir, exist_ok=True)

		wavelet_results_file_path = wavelet_analysis_dir + f"/{volume['htid']}_wavelet_results.csv"
		full_wavelet_results.to_csv(wavelet_results_file_path, index=False)

		skipped_dwts_file_path = wavelet_analysis_dir + f"/{volume['htid']}_skipped_dwts.csv"
		if len(dwt_skipped_results) > 0:
			dwt_skipped_results.to_csv(skipped_dwts_file_path, index=False)

		skipped_cwts_file_path = wavelet_analysis_dir + f"/{volume['htid']}_skipped_cwts.csv"
		if len(cwt_skipped_results) > 0:
			cwt_skipped_results.to_csv(skipped_cwts_file_path, index=False)


	# Create DataFrame from volume frequencies
	volume_frequencies_df = pd.DataFrame(volume_frequencies)
	
	if len(altair_charts) > 0:
		# Save Altair charts as images
		combined_charts = alt.vconcat(*altair_charts)
		# Save the chart
		save_chart(combined_charts, f"{output_dir}/annotated_tokens_per_page/{periodical_name}_tokens_per_page_chart.png", scale_factor=2.0)

	# Calculate consensus issue length based on median dominant frequency
	volume_frequencies_df['raw_consensus_issue_length'] = volume_frequencies_df['raw_dynamic_cutoff'].median()
	volume_frequencies_df['smoothed_consensus_issue_length'] = volume_frequencies_df['smoothed_dynamic_cutoff'].median()
	volume_frequencies_df['raw_consensus_issue_length'] = volume_frequencies_df['raw_consensus_issue_length'].fillna(0)
	volume_frequencies_df['smoothed_consensus_issue_length'] = volume_frequencies_df['smoothed_consensus_issue_length'].fillna(0)

	plot_volume_frequencies_matplotlib(volume_frequencies, periodical_name, output_dir)
	plot_tokens_per_page(volume_frequencies, output_dir, periodical_name)

	return volume_frequencies_df

def generate_token_frequency_analysis(should_filter_greater_than_numbers: bool, should_filter_implied_zeroes: bool, only_use_annotated_periodicals: bool, rerun_code: bool = False):
	"""
	Generate token frequency analysis for all identified HathiTrust periodicals.

	Parameters:
	- should_filter_greater_than_numbers: Flag indicating whether to filter out numbers greater than the max possible page number.
	- should_filter_implied_zeroes: Flag indicating whether to filter out implied zeroes.
	- only_use_annotated_periodicals: Flag indicating whether to process only annotated periodicals.
	- rerun_code: Flag indicating whether to rerun the code.
	"""

	# Count the number of matching files
	matching_files = []
	for directory, _, files in tqdm(os.walk("../datasets/annotated_ht_ef_datasets/"), desc="Counting matching files"):
		for file in files:
			if file.endswith(".csv") and 'individual' in file:
				if os.path.exists(os.path.join(directory, file)):
					publication_name = directory.split("/")[-2]
					volume_number = directory.split("/")[-1]
					matching_files.append({"file": file, "directory": directory, "file_path": os.path.join(directory, file), "periodical_title": publication_name, "volume_directory": volume_number})
	matching_files_df = pd.DataFrame(matching_files)
	console.print(f"Found {len(matching_files_df)} matching files.", style="bright_green")

	volume_features_output_path = os.path.join("..", "datasets", "all_volume_features_and_frequencies.csv")
	volume_features_exist = False
	if os.path.exists(volume_features_output_path) and rerun_code:
		volume_features_df = read_csv_file(volume_features_output_path)
		volume_features_exist = True
		console.print(f"Found {len(volume_features_df)} existing volume features.", style="bright_green")
	elif os.path.exists(volume_features_output_path) and not rerun_code:
		#delete the file
		os.remove(volume_features_output_path)
		volume_features_df = pd.DataFrame()
	else:
		volume_features_df = pd.DataFrame()

	data_directory_path = get_data_directory_path()
	console.print(f"Reading preidentified periodicals from {data_directory_path}..", style="bright_blue")
	
	preidentified_periodicals_df = read_csv_file(os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", "periodical_metadata", "classified_preidentified_periodicals_with_full_metadata.csv"))
	periodical_titles = preidentified_periodicals_df['lowercase_periodical_name'].unique()

	# Process only annotated periodicals if specified
	for index, title in enumerate(tqdm(periodical_titles, desc="Processing periodicals")):
		console.print(f"Processing periodical {title} number ({index + 1} out of {len(periodical_titles)})..", style="bright_blue")
		subset_preidentified_periodicals_df = preidentified_periodicals_df[preidentified_periodicals_df['lowercase_periodical_name'] == title]
		volumes = subset_preidentified_periodicals_df.volume_directory.unique()
		subset_matching_files_df = matching_files_df[matching_files_df['volume_directory'].isin(volumes)]

		# Skip unannotated periodicals if the flag is set
		if only_use_annotated_periodicals and len(subset_matching_files_df) == 0:
			console.print(f"No annotated files found for periodical {title}. Skipping...", style="bright_red")
			continue

		
		volume_paths = []

		for _, row in subset_preidentified_periodicals_df[subset_preidentified_periodicals_df.volume_directory.notna()].iterrows():
			if volume_features_exist:
				processed_htid = row.volume_directory.replace("_", ".")
				volume_in_features = volume_features_df[volume_features_df['htid'] == processed_htid].copy()
				if (len(volume_in_features) > 0) and (not rerun_code):
					console.print(f"Volume {row.volume_directory} already exists in volume features..", style="bright_yellow")
					continue

			matched_row = subset_matching_files_df[subset_matching_files_df['volume_directory'] == row.volume_directory]

			# Skip unannotated volumes if the flag is set
			if only_use_annotated_periodicals and len(matched_row) == 0:
				console.print(f"Volume {row.volume_directory} is not annotated. Skipping...", style="bright_red")
				continue
			
			is_annotated_periodical = len(matched_row) > 0
			file_path = matched_row.file_path.values[0] if len(matched_row) > 0 else os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", row.publication_directory, "volumes", row['volume_directory'], row['volume_directory'] + "_individual_tokens.csv")
			volume_paths.append({
				'file_path': file_path,
				'is_annotated_periodical': is_annotated_periodical,
				'should_filter_greater_than_numbers': should_filter_greater_than_numbers,
				'should_filter_implied_zeroes': should_filter_implied_zeroes,
				'table_row_index': row['table_row_index'],
				'lowercase_periodical_name': row['lowercase_periodical_name'],
				'htid': row['htid'],
				'volume_classification': row['volume_classification'],
				'title_classification': row['title_classification'],
			})

		# If no volumes found, skip this periodical
		if len(volume_paths) == 0:
			console.print(f"No valid volumes found for periodical {title}. Skipping...", style="bright_red")
			continue

		volume_paths_df = pd.DataFrame(volume_paths)
		volume_frequencies = generate_volume_embeddings(volume_paths_df, output_dir="../figures")
		# Drop amplitutde and frequency columns for saving file space
		volume_frequencies = volume_frequencies.drop(columns=['raw_positive_frequencies', 'raw_positive_amplitudes', 'smoothed_positive_frequencies', 'smoothed_positive_amplitudes', 'tokens_per_page', 'page_numbers', 'digits_per_page'])
		# Save volume frequencies to CSV
		if os.path.exists(volume_features_output_path):
			volume_frequencies.to_csv(volume_features_output_path, mode='a', index=False, header=False)
		else:
			volume_frequencies.to_csv(volume_features_output_path, index=False)

if __name__ == "__main__":
	filter_greater_than_numbers = True
	filter_implied_zeroes = True
	should_rerun_code = False
	should_only_use_annotated_periodicals = False
	generate_token_frequency_analysis(filter_greater_than_numbers, filter_implied_zeroes,  should_only_use_annotated_periodicals, should_rerun_code)