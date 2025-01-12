# Standard library imports
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import altair as alt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
from rich.console import Console
import pywt
from scipy.fft import fft
from scipy.signal import find_peaks
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr

# Disable max rows for Altair
alt.data_transformers.disable_max_rows()

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

## WAVELET SIGNAL ANALYSIS CODE

def process_wavelet_results(results: list, skipped_results: list, signal_type: str, metrics: list = ['wavelet_energy_entropy', 'wavelet_sparsity']) -> tuple:
	"""
	Process and clean wavelet results, handling infinite or NaN values and combining skipped results.

	Parameters:
	-----------
	results : list
		List of wavelet analysis results.
	skipped_results : list
		List of skipped wavelet results.
	signal_type : str
		Type of signal being analyzed (e.g., "DWT", "CWT", "SWT").
	metrics : list
		List of metrics to check for missing or invalid values.

	Returns:
	--------
	tuple:
		- cleaned_results_df: pd.DataFrame
		  DataFrame containing cleaned wavelet analysis results.
		- combined_skipped_results_df: pd.DataFrame
		  DataFrame containing combined skipped and error results.
	"""
	total_results = pd.DataFrame(results)

	# Check if the specified metrics columns are in the DataFrame
	missing_metrics = [metric for metric in metrics if metric not in total_results.columns]
	if missing_metrics:
		console.print(f"[yellow]Warning: The following metrics are missing from the results and will be excluded from the analysis: {missing_metrics}[/yellow]")
		metrics = [metric for metric in metrics if metric in total_results.columns]
	# Replace inf values and drop rows with missing metrics
	cleaned_results_df = total_results.replace([np.inf, -np.inf], np.nan).dropna(subset=metrics).reset_index(drop=True)if len(metrics) > 0 else total_results
		
	console.print(f"Total valid results for {signal_type}: {len(cleaned_results_df)}", style="bright_green")

	# Combine skipped results and invalid rows
	error_results = total_results[~total_results.index.isin(cleaned_results_df.index)]
	skipped_results_df = pd.DataFrame(skipped_results)
	combined_skipped_results_df = pd.concat([error_results, skipped_results_df], ignore_index=True)
	console.print(f"Total skipped results for {signal_type}: {len(combined_skipped_results_df)}", style="bright_red")
	if len(cleaned_results_df) == 0:
		console.print(f"[bright_red]No valid results found for {signal_type}.[/bright_red]")
		console.print(f"First row error: {combined_skipped_results_df.iloc[0]['error']}")
	return cleaned_results_df, combined_skipped_results_df

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

def adaptive_threshold(coeffs: list) -> float:
	"""
	Calculate an adaptive threshold based on the Median Absolute Deviation (MAD).

	Parameters:
	----------
	coeffs : list of np.ndarray
		Wavelet decomposition coefficients.

	Returns:
	--------
	threshold : float
		Adaptive threshold for sparsity computation.
	"""
	flat_coeffs = np.abs(np.concatenate(coeffs))  # Flatten coefficients and take absolute values
	mad = np.median(np.abs(flat_coeffs - np.median(flat_coeffs)))  # Compute MAD
	return mad * 1.4826  # Scale factor for Gaussian distribution

def adaptive_sparsity_measure(coeffs: list) -> tuple:
	"""
	Measure sparsity of wavelet coefficients using an adaptive threshold.

	Parameters:
	-----------
	coeffs : list of np.ndarray
		Wavelet decomposition coefficients.

	Returns:
	--------
	sparsity : float
		Percentage of near-zero coefficients based on adaptive threshold.
	threshold : float
		Computed adaptive threshold for sparsity measurement.
	"""
	flat_coeffs = np.abs(np.concatenate(coeffs))
	
	# Calculate adaptive threshold using Median Absolute Deviation (MAD)
	threshold = np.median(flat_coeffs) * 1.4826  # MAD scaling for normal distribution
	
	# Compute sparsity as the percentage of coefficients below the adaptive threshold
	sparsity = np.sum(flat_coeffs < threshold) / len(flat_coeffs)
	
	return sparsity, threshold

def wavelet_entropy(coeffs: list) -> float:
	"""
	Calculate wavelet entropy as a measure of signal complexity.

	Parameters:
	-----------
	coeffs : list of np.ndarray
		Wavelet decomposition coefficients.
	
	Returns:
	-----------
	entropy : float
		Wavelet entropy.
	"""
	magnitudes = np.abs(coeffs)
	total_energy = np.sum(magnitudes ** 2)
	probabilities = (magnitudes ** 2) / (total_energy + 1e-12)
	return -np.sum(probabilities * np.log2(probabilities + 1e-12))

def signal_smoothness(signal: np.ndarray) -> float:
	"""
	Compute signal smoothness based on second-order differences.

	Parameters:
	-----------
	signal : np.ndarray
		Signal to analyze.

	Returns:
	-----------
	smoothness : float
		Smoothness measure based on second-order differences.
	"""
	second_derivative = np.diff(signal, n=2)
	smoothness = 1 / (1 + np.mean(second_derivative ** 2))
	return smoothness

def correlation_coefficients(original: np.ndarray, reconstructed: np.ndarray) -> float:
	"""
	Calculate the correlation coefficient between the original and reconstructed signals.

	Parameters:
	-----------
	original : np.ndarray
		The original signal.
	reconstructed : np.ndarray
		The reconstructed signal.

	Returns:
	-----------
	correlation : float
		Correlation coefficient.
	"""
	return np.corrcoef(original, reconstructed)[0, 1]

def signal_variance_across_levels(coeffs: list) -> list:
	"""
	Calculate the variance of wavelet coefficients across decomposition levels.

	Parameters:
	-----------
	coeffs : list of np.ndarray
		Variance of coefficients across levels.

	Returns:
	-----------
	variances : list
		Variance of coefficients across levels.
	"""
	return [np.var(c) for c in coeffs]

def compute_additional_wavelet_features(coeffs: list, reconstructed_signal: np.ndarray, original_signal: np.ndarray) -> dict:
	"""
	Compute additional features from wavelet coefficients and reconstructed signal.
	
	Parameters:
	----------
	coeffs : list or np.ndarray
		Wavelet decomposition coefficients.
	reconstructed_signal : np.ndarray
		Signal reconstructed from the wavelet coefficients.
	original_signal : np.ndarray
		The original signal used for decomposition.

	Returns:
	--------
	features : dict
		Dictionary containing additional computed features.
	"""
	entropy = wavelet_entropy(coeffs)
	variances = signal_variance_across_levels(coeffs)
	avg_variance = np.mean(variances)
	variance_ratio = max(variances) / (sum(variances) + 1e-12)
	smoothness = signal_smoothness(reconstructed_signal)
	correlation = correlation_coefficients(original_signal, reconstructed_signal)

	return {
		"wavelet_entropy": entropy,
		"avg_variance_across_levels": avg_variance,
		"variance_ratio_across_levels": variance_ratio,
		"smoothness": smoothness,
		"correlation": correlation,
	}

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
	# Ensure signal is valid
	if tokens_signal is None or len(tokens_signal) == 0:
		console.print(f"[bright_red]Error: Empty or invalid signal for {use_signal_type}.[/bright_red]")
		return {}

	# Clean the signal to handle NaN or Inf values
	tokens_signal = np.nan_to_num(tokens_signal, nan=0.0, posinf=0.0, neginf=0.0)

	try:
		# Perform FFT for frequency analysis
		tokens_fft = fft(tokens_signal)
		frequencies = np.fft.fftfreq(len(tokens_fft))

		# Analyze positive frequencies and amplitudes
		positive_frequencies = frequencies[:len(frequencies) // 2]
		positive_amplitudes = np.abs(tokens_fft[:len(tokens_fft) // 2])

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
		prominence = prominence or np.std(tokens_signal) * 0.1
		distance = distance or max(1, len(tokens_signal) // 20)
		relative_peaks, relative_properties = find_peaks(tokens_signal, prominence=prominence, distance=distance)
		relative_num_peaks = len(relative_peaks)
		avg_prominence = np.mean(relative_properties["prominences"]) if relative_num_peaks > 0 else None

		# Calculate autocorrelation
		autocorr = np.correlate(tokens_signal, tokens_signal, mode='full')
		max_autocorr = np.max(autocorr[len(autocorr) // 2:])

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
			console.print(f"Signal Length: {len(tokens_signal)}")
			console.print(f"Signal Mean: {np.mean(tokens_signal):.2f}, Std: {np.std(tokens_signal):.2f}")
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

## DWT, CWT, and SWT EVALUATION FUNCTIONS

def pad_signal(signal: np.ndarray, max_level: int) -> tuple:
	"""
	Pad the signal to ensure it has an even length and is compatible with SWT.

	Parameters:
	-----------
	signal : np.ndarray
		Input signal to pad.
	max_level : int
		Maximum decomposition level for SWT.

	Returns:
	--------
	tuple:
	- padded_signal : np.ndarray
		Signal padded to the nearest valid length.
	- is_padded: bool
		A flag of whether the signal was padded.
	"""
	# Calculate the required length
	min_length = 2 ** max_level
	if len(signal) % 2 != 0 or len(signal) < min_length:
		# Make the signal length even and divisible by 2^max_level
		pad_length = max(0, min_length - len(signal))  # Ensure length >= 2^max_level
		padded_signal = np.pad(signal, (0, pad_length), mode="constant", constant_values=0)
		if len(padded_signal) % 2 != 0:
			padded_signal = np.pad(padded_signal, (0, 1), mode="constant", constant_values=0)
		console.print(
			f"[yellow]Padded signal from {len(signal)} to {len(padded_signal)} to meet SWT requirements.[/yellow]"
		)
		return padded_signal, True
	return signal, False

def validate_max_level(signal_length: int, requested_max_level: int) -> tuple:
	"""
	Validate and adjust the maximum level for SWT based on the signal length.

	Parameters:
	-----------
	signal_length : int
		Length of the input signal.
	requested_max_level : int
		Requested maximum decomposition level.

	Returns:
	--------
	tuple:
	- validated_max_level : int
		Adjusted maximum level based on the signal length.
	- is_original_level: bool
		Indicates whether the original requested max level was valid.
	"""
	max_possible_level = pywt.swt_max_level(signal_length)
	validated_max_level = min(requested_max_level, max_possible_level)
	is_original_level = (requested_max_level == validated_max_level)
	return validated_max_level, is_original_level

def is_wavelet_compatible(signal_length: int, wavelet: str, level: int) -> bool:
	"""
	Check if the wavelet is compatible with the signal length at the given level.

	Parameters:
	-----------
	signal_length : int
		Length of the signal.
	wavelet : str
		Name of the wavelet.
	level : int
		Decomposition level.

	Returns:
	--------
	bool
		True if the wavelet is compatible, False otherwise.
	"""
	try:
		wavelet_filter_len = pywt.Wavelet(wavelet).dec_len
		max_level = pywt.dwt_max_level(signal_length, wavelet_filter_len)
		return level <= max_level
	except ValueError:
		return False

def determine_scales(signal_length: int, max_scale: int = 128, dynamic: bool = True) -> np.ndarray:
	"""
	Determine wavelet scales dynamically based on signal length. This is useful for CWT.

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

def process_dwt_wavelet(signal: np.ndarray, wavelet: str, modes: list, signal_type: str) -> tuple:
	"""
	Process a single DWT wavelet for the given signal and modes.

	Parameters:
	-----------
	signal : np.ndarray
		The signal to analyze.
	wavelet : str
		The wavelet to use for decomposition.
	modes : list
		List of signal extension modes to test.
	signal_type : str
		Type of signal either raw or smoothed.

	Returns:
	--------
	tuple:
		- results: List of dictionaries containing wavelet analysis results.
		- skipped_wavelets: List of dictionaries containing skipped wavelets and errors.
	"""
	results = []
	skipped_wavelets = []
	try:
		wavelet_filter_len = pywt.Wavelet(wavelet).dec_len
		if len(signal) < wavelet_filter_len:
			raise ValueError(f"Signal is too short for wavelet {wavelet}")

		max_level = pywt.dwt_max_level(len(signal), wavelet_filter_len)
		for level in range(1, max_level + 1):
			for mode in modes:
				# Check compatibility
				if not is_wavelet_compatible(len(signal), wavelet, level):
					skipped_wavelets.append({
						'wavelet': wavelet,
						'wavelet_level': level,
						'wavelet_mode': mode,
						'error': f"Incompatible wavelet or decomposition level for signal length {len(signal)}.",
						'signal_length': len(signal)
					})
					continue

				try:
					coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)
					reconstructed_signal = pywt.waverec(coeffs, wavelet, mode=mode)[:len(signal)]

					mse = np.mean((signal - reconstructed_signal) ** 2)
					psnr_value = psnr(signal, reconstructed_signal, data_range=np.max(signal) - np.min(signal))
					energy_entropy = energy_entropy_ratio(coeffs)
					sparsity, threshold = adaptive_sparsity_measure(coeffs)
					additional_features = compute_additional_wavelet_features(coeffs, reconstructed_signal, signal)
					emd_value = wasserstein_distance(signal, reconstructed_signal)
					kl_div_value = sum(rel_entr(signal, reconstructed_signal + 1e-12))

					results.append({
						'wavelet': wavelet,
						'wavelet_level': level,
						'wavelet_mode': mode,
						'wavelet_mse': mse,
						'wavelet_psnr': psnr_value,
						'wavelet_energy_entropy': energy_entropy,
						'wavelet_sparsity': sparsity,
						'wavelet_adaptive_threshold': threshold,
						'signal_length': len(signal),
						'signal_type': signal_type,
						'emd_value': emd_value,
						'kl_divergence': kl_div_value,
						**additional_features
					})
				except Exception as e:
					skipped_wavelets.append({
						'wavelet': wavelet,
						'wavelet_level': level,
						'wavelet_mode': mode,
						'error': str(e),
						'signal_length': len(signal)
					})
	except Exception as e:
		skipped_wavelets.append({'wavelet': wavelet, 'error': str(e)})

	return results, skipped_wavelets

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
	total_results = []
	skipped_results = []

	for wavelet in tqdm(wavelets, desc=f"Testing DWT Wavelets for {signal_type}"):
		wavelet_results, wavelet_skipped = process_dwt_wavelet(signal, wavelet, modes, signal_type)
		total_results.extend(wavelet_results)
		skipped_results.extend(wavelet_skipped)

	cleaned_total_results, cleaned_skipped_results = process_wavelet_results(pd.DataFrame(total_results), pd.DataFrame(skipped_results), signal_type)
	return cleaned_total_results, cleaned_skipped_results

def evaluate_dwt_performance_parallel(signal: np.ndarray, wavelets: list, modes: list, signal_type: str) -> tuple:
	"""
	Parallelized version of DWT evaluation.

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
	total_results = []
	skipped_results = []
	max_workers = min(len(wavelets), multiprocessing.cpu_count() - 1)

	console.print(f"Using {max_workers} workers for parallel DWT processing.", style="bright_cyan")

	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = [
			executor.submit(process_dwt_wavelet, signal, wavelet, modes, signal_type) for wavelet in wavelets
		]
		for future in tqdm(as_completed(futures), total=len(futures), desc=f"Testing DWT Wavelets for {signal_type}"):
			try:
				wavelet_results, wavelet_skipped = future.result()
				total_results.extend(wavelet_results)
				skipped_results.extend(wavelet_skipped)
			except Exception as e:
				console.print(f"Unexpected error during parallel processing: {e}", style="bright_red")

	cleaned_total_results, cleaned_skipped_results = process_wavelet_results(pd.DataFrame(total_results), pd.DataFrame(skipped_results), signal_type)
	return cleaned_total_results, cleaned_skipped_results

def process_cwt_wavelet(signal: np.ndarray, wavelet: str, scales: np.ndarray, signal_type: str) -> tuple:
	"""
	Process a single wavelet for Continuous Wavelet Transform (CWT).

	Parameters:
	-----------
	signal : np.ndarray
		Signal to analyze.
	wavelet : str
		Wavelet to use for decomposition.
	scales : np.ndarray
		Array of scales for the wavelet transform.
	signal_type : str
		The type of signal being analyzed ('raw' or 'smoothed').

	Returns:
	--------
	tuple:
		- results: List of dictionaries containing wavelet analysis results.
		- skipped_results: List of dictionaries containing skipped wavelets and errors.
	"""
	results = []
	skipped_results = []
	try:
		# Perform Continuous Wavelet Transform
		coeffs, _ = pywt.cwt(signal, scales=scales, wavelet=wavelet)

		# Compute reconstructed signal (summed across scales)
		reconstructed_signal = np.sum(coeffs, axis=0)

		# Compute PSNR
		psnr_value = psnr(signal, reconstructed_signal, data_range=np.max(signal) - np.min(signal))

		# Compute Metrics
		total_energy = np.sum(coeffs ** 2)
		entropy = -np.sum(coeffs ** 2 / total_energy * np.log2(coeffs ** 2 / total_energy + 1e-12), axis=None)
		energy_entropy = total_energy / entropy if entropy > 0 else np.inf

		# Use adaptive sparsity measure
		sparsity, adaptive_threshold = adaptive_sparsity_measure(coeffs)

		# Compute additional features
		additional_features = compute_additional_wavelet_features(coeffs, reconstructed_signal, signal)

		# Calculate KL Divergence and EMD
		emd_value = wasserstein_distance(signal, reconstructed_signal)
		kl_div_value = sum(rel_entr(signal, reconstructed_signal))
		# Append Results
		results.append({
			'signal_type': signal_type,
			'wavelet': wavelet,
			'wavelet_energy_entropy': energy_entropy,
			'wavelet_sparsity': sparsity,
			'wavelet_adaptive_threshold': adaptive_threshold,
			'wavelet_psnr': psnr_value,
			'signal_length': len(signal),
			'scales_used': len(scales),
			'emd_value': emd_value,
			'kl_divergence': kl_div_value,
			**additional_features
		})
	except Exception as e:
		skipped_results.append({'wavelet': wavelet, 'error': str(e), 'signal_length': len(signal), 'signal_type': signal_type, 'scales_used': len(scales)})

	return results, skipped_results

def evaluate_cwt_performance(signal: np.ndarray, wavelets: list, signal_type: str, max_scale: int = 128, dynamic_scales: bool = True) -> tuple:
	"""
	Evaluate Continuous Wavelet Transform (CWT) using MSE, energy-to-entropy ratio, and sparsity for a given signal type.

	Parameters:
	-----------
	signal : np.ndarray
		Signal to analyze.
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
	tuple:
		- total_results: pd.DataFrame
		  DataFrame containing wavelet analysis results.
		- skipped_results: pd.DataFrame
		  DataFrame containing skipped wavelets.
	"""
	scales = determine_scales(len(signal), max_scale=max_scale, dynamic=dynamic_scales)
	total_results = []
	skipped_results = []

	for wavelet in tqdm(wavelets, desc=f"Testing CWT Wavelets for {signal_type}"):
		wavelet_results, wavelet_skipped = process_cwt_wavelet(signal, wavelet, scales, signal_type)
		total_results.extend(wavelet_results)
		skipped_results.extend(wavelet_skipped)

	cleaned_total_results, cleaned_skipped_results = process_wavelet_results(pd.DataFrame(total_results), pd.DataFrame(skipped_results), signal_type)
	return cleaned_total_results, cleaned_skipped_results

def evaluate_cwt_performance_parallel(signal: np.ndarray, wavelets: list, signal_type: str, max_scale: int = 128, dynamic_scales: bool = True) -> tuple:
	"""
	Parallelized evaluation of Continuous Wavelet Transform (CWT).

	Parameters:
	-----------
	signal : np.ndarray
		Signal to analyze.
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
	tuple:
		- total_results: pd.DataFrame
		  DataFrame containing wavelet analysis results.
		- skipped_results: pd.DataFrame
		  DataFrame containing skipped wavelets.
	"""
	scales = determine_scales(len(signal), max_scale=max_scale, dynamic=dynamic_scales)
	total_results = []
	skipped_results = []
	max_workers = min(len(wavelets), multiprocessing.cpu_count() - 1)

	console.print(f"Using {max_workers} workers for parallel CWT processing.", style="bright_cyan")

	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = [
			executor.submit(process_cwt_wavelet, signal, wavelet, scales, signal_type) for wavelet in wavelets
		]
		for future in tqdm(as_completed(futures), total=len(futures), desc=f"Testing CWT Wavelets for {signal_type}"):
			try:
				wavelet_results, wavelet_skipped = future.result()
				total_results.extend(wavelet_results)
				skipped_results.extend(wavelet_skipped)
			except Exception as e:
				console.print(f"Unexpected error during parallel processing: {e}", style="bright_red")

	cleaned_total_results, cleaned_skipped_results = process_wavelet_results(pd.DataFrame(total_results), pd.DataFrame(skipped_results), signal_type)
	return cleaned_total_results, cleaned_skipped_results

def process_swt_wavelet(signal: np.ndarray, wavelet: str, signal_type: str, max_level: int) -> tuple:
	"""
	Process a single wavelet for Stationary Wavelet Transform (SWT).

	Parameters:
	-----------
	signal : np.ndarray
		Signal to analyze.
	wavelet : str
		Wavelet to use for decomposition.
	max_level : int
		The maximum decomposition level for SWT.
	signal_type : str
		The type of signal being analyzed ('raw' or 'smoothed').

	Returns:
	--------
	tuple:
	- results: List of dictionaries containing wavelet analysis results.
	- skipped_results: List of dictionaries containing skipped wavelets and errors.
	"""
	results = []
	skipped_results = []

	try:
		# Validate and adjust max_level
		max_level, is_original_level = validate_max_level(len(signal), max_level)
		if max_level < 1:
			skipped_results.append({
				'wavelet': wavelet,
				'error': f"Max level too low for wavelet {wavelet} and signal length {len(signal)}.",
				'signal_length': len(signal),
				'signal_type': signal_type,
				'decomposition_levels': max_level
			})
			raise ValueError(f"Max level too low for wavelet {wavelet} and signal length {len(signal)}.")

		# Pad the signal to meet SWT requirements
		padded_signal, is_padded = pad_signal(signal, max_level)

		# Decompose signal using SWT
		coeffs = pywt.swt(padded_signal, wavelet=wavelet, level=max_level, start_level=0)
		approx_coeffs, detail_coeffs = zip(*coeffs)

		# Reconstruct and trim signal to original length
		reconstructed_signal = np.sum([approx_coeffs[-1]] + list(detail_coeffs), axis=0)[:len(signal)]

		# Compute PSNR
		wavelet_psnr = psnr(signal, reconstructed_signal, data_range=np.max(signal) - np.min(signal))

		# Compute Metrics
		total_energy = np.sum([np.sum(np.array(c) ** 2) for c in detail_coeffs])
		entropy = -np.sum(
			[
				np.sum(
					np.array(c) ** 2 / total_energy
					* np.log2(np.array(c) ** 2 / total_energy + 1e-12)
				)
				for c in detail_coeffs
			]
		)
		energy_entropy = total_energy / (entropy if entropy > 0 else np.inf)

		# Use adaptive sparsity measure
		sparsity, adaptive_threshold = adaptive_sparsity_measure(detail_coeffs)

		# Compute additional features
		additional_features = compute_additional_wavelet_features(coeffs, reconstructed_signal, signal)

		# Calculate KL Divergence and EMD
		emd_value = wasserstein_distance(signal, reconstructed_signal)
		kl_div_value = sum(rel_entr(signal, reconstructed_signal))

		# Append results
		results.append({
			'signal_type': signal_type,
			'wavelet': wavelet,
			'wavelet_energy_entropy': energy_entropy,
			'wavelet_sparsity': sparsity,
			'wavelet_adaptive_threshold': adaptive_threshold,
			'wavelet_psnr': wavelet_psnr,
			'signal_length': len(signal),
			'decomposition_levels': max_level,
			'original_level': is_original_level,
			'padded': is_padded,
			'emd_value': emd_value,
			'kl_divergence': kl_div_value,
			**additional_features
		})

	except Exception as e:
		skipped_results.append({
			'wavelet': wavelet,
			'error': str(e),
			'signal_length': len(signal),
			'signal_type': signal_type,
			'decomposition_levels': max_level
		})

	return results, skipped_results

def evaluate_swt_performance(signal: np.ndarray, wavelets: list, signal_type: str, max_level: int = 5) -> tuple:
	"""
	Evaluate Stationary Wavelet Transform (SWT) using adaptive sparsity, energy-to-entropy ratio, PSNR, and additional features for a given signal type.

	Parameters:
	-----------
	signal : np.ndarray
		Signal to analyze.
	wavelets : list of str
		List of wavelet names to test.
	signal_type : str
		The type of signal being analyzed ('raw' or 'smoothed').
	max_level : int
		Maximum decomposition level for SWT.

	Returns:
	--------
	tuple:
		- total_results: pd.DataFrame
		  DataFrame containing wavelet analysis results.
		- skipped_results: pd.DataFrame
		  DataFrame containing skipped wavelets.
	"""
	total_results = []
	skipped_results = []

	for wavelet in tqdm(wavelets, desc=f"Testing SWT Wavelets for {signal_type}"):
		wavelet_results, wavelet_skipped = process_swt_wavelet(signal, wavelet, signal_type, max_level)
		total_results.extend(wavelet_results)
		skipped_results.extend(wavelet_skipped)

	cleaned_total_results, cleaned_skipped_results = process_wavelet_results(
		pd.DataFrame(total_results), pd.DataFrame(skipped_results), signal_type
	)
	return cleaned_total_results, cleaned_skipped_results

def evaluate_swt_performance_parallel(signal: np.ndarray, wavelets: list, signal_type: str, max_level: int = 5) -> tuple:
	"""
	Parallelized evaluation of Stationary Wavelet Transform (SWT).

	Parameters:
	-----------
	signal : np.ndarray
		Signal to analyze.
	wavelets : list of str
		List of wavelet names to test.
	signal_type : str
		The type of signal being analyzed ('raw' or 'smoothed').
	max_level : int
		Maximum decomposition level for SWT.

	Returns:
	--------
	tuple:
		- total_results: pd.DataFrame
		  DataFrame containing wavelet analysis results.
		- skipped_results: pd.DataFrame
		  DataFrame containing skipped wavelets.
	"""
	total_results = []
	skipped_results = []
	max_workers = min(len(wavelets), multiprocessing.cpu_count() - 1)

	console.print(f"Using {max_workers} workers for parallel SWT processing.", style="bright_cyan")

	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = [
			executor.submit(process_swt_wavelet, signal, wavelet, signal_type, max_level) for wavelet in wavelets
		]
		for future in tqdm(as_completed(futures), total=len(futures), desc=f"Testing SWT Wavelets for {signal_type}"):
			try:
				wavelet_results, wavelet_skipped = future.result()
				total_results.extend(wavelet_results)
				skipped_results.extend(wavelet_skipped)
			except Exception as e:
				console.print(f"Unexpected error during parallel processing: {e}", style="bright_red")

	cleaned_total_results, cleaned_skipped_results = process_wavelet_results(
		pd.DataFrame(total_results), pd.DataFrame(skipped_results), signal_type
	)
	return cleaned_total_results, cleaned_skipped_results
