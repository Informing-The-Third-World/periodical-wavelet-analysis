# Standard library imports
import warnings
import sys

# Third-party imports
import pandas as pd
import numpy as np
import altair as alt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
from rich.console import Console
import pywt
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
from skimage.metrics import peak_signal_noise_ratio as psnr

# Local application imports
sys.path.append("..")
from scripts.wavelet_scripts.generate_wavelet_features import *

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
