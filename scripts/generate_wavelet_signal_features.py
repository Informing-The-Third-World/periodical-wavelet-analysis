# Standard library imports
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import altair as alt
from rich.console import Console
from scipy.fft import fft
from scipy.signal import find_peaks

# Disable max rows for Altair
alt.data_transformers.disable_max_rows()

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

## CALCULATE WAVELET FEATURES
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

def calculate_fft(
	tokens_signal: np.ndarray,
	verbose: bool,
	min_length: int = 16,
	snr_thresholds: list = [3.0, 5.0, 7.0],  # Now iterating over SNR thresholds
	stationarity_thresholds: list = [0.3, 0.5, 0.7]  # Iterating over stationarity thresholds
) -> tuple:
	"""
	Calculate the Fast Fourier Transform (FFT) of a given signal. This function first ensures the signal is suitable for FFT analysis
	by validating its length, signal-to-noise ratio (SNR), and stationarity. If the signal passes these checks, the FFT is performed.

	Parameters:
	-----------
	tokens_signal : np.ndarray
		The signal to analyze.
	verbose : bool
		Whether to display verbose output.
	min_length : int, optional
		Minimum length of the signal for FFT analysis. Default is 16.
	snr_thresholds : list, optional
		List of signal-to-noise ratio (SNR) thresholds to test.
	stationarity_thresholds : list, optional
		List of stationarity thresholds to check power spectral density variations.

	Returns:
	--------
	tuple:
		- positive_amplitudes: np.ndarray or None
		  Positive amplitudes of the signal if valid, otherwise None.
		- positive_frequencies: np.ndarray or None
		  Positive frequencies of the signal if valid, otherwise None.
		- final_snr_threshold: float or None
		  The final SNR threshold used for FFT analysis.
		- final_stationarity_threshold: float or None
		  The final stationarity threshold used for FFT analysis.
	"""
	# Check signal length
	if len(tokens_signal) < min_length:
		if verbose:
			console.print(f"[yellow]Signal length ({len(tokens_signal)}) is too short for meaningful FFT analysis.[/yellow]")
		return None, None, None, None  # Return all None if FFT is not possible

	# Iterate over SNR and stationarity thresholds
	for snr_threshold in snr_thresholds:
		for stationarity_threshold in stationarity_thresholds:
			# Compute Signal-to-Noise Ratio (SNR)
			smoothed_signal = np.convolve(tokens_signal, np.ones(5) / 5, mode='same')
			signal_power = np.sum(tokens_signal ** 2)
			noise_power = np.sum((tokens_signal - smoothed_signal) ** 2)
			snr = 10 * np.log10(signal_power / (noise_power + 1e-6))  # Avoid division by zero

			if snr < snr_threshold:
				if verbose:
					console.print(f"[yellow]Low SNR ({snr:.2f} dB); FFT may not provide meaningful results (SNR threshold={snr_threshold}).[/yellow]")
				continue  # Try the next combination

			# Check stationarity
			chunk_size = max(len(tokens_signal) // 4, 16)  # Ensure minimum chunk size
			psd_variations = []
			for i in range(0, len(tokens_signal), chunk_size):
				chunk = tokens_signal[i:i + chunk_size]
				chunk_fft = np.abs(fft(chunk))[:len(chunk) // 2]  # Positive amplitudes only
				psd_variations.append(chunk_fft.sum())

			# Add safeguard for std deviation (prevent division issues)
			psd_variations_std = np.std(psd_variations)
			psd_variations_mean = np.mean(psd_variations) + 1e-6  # Small epsilon to prevent divide by zero
			if psd_variations_std > psd_variations_mean * stationarity_threshold:
				if verbose:
					console.print(f"[yellow]Signal shows significant non-stationarity; FFT results may be unreliable (Threshold={stationarity_threshold}).[/yellow]")
				continue  # Try the next combination

			# Perform FFT
			tokens_fft = fft(tokens_signal)
			frequencies = np.fft.fftfreq(len(tokens_fft))
			positive_frequencies = frequencies[:len(frequencies) // 2]
			positive_amplitudes = np.abs(tokens_fft[:len(tokens_fft) // 2])

			# Return the first successful configuration
			return positive_amplitudes, positive_frequencies, snr_threshold, stationarity_threshold

	# If all iterations failed
	return None, None, None, None

def analyze_fft_peaks(tokens_fft: np.ndarray, frequencies: np.ndarray, verbose: bool, min_peak_prominence: float = 0.01) -> dict:
	"""
	Analyze positive frequencies and amplitudes from FFT results to determine key characteristics. If either tokens_fft or frequencies is None, peak analysis is skipped.

	Parameters:
	-----------
	tokens_fft : np.ndarray
		FFT-transformed signal (complex values).
	frequencies : np.ndarray
		Frequencies corresponding to the FFT-transformed signal.
	min_peak_prominence : float, optional
		Minimum prominence of peaks for detection. Default is 0.01.

	Returns:
	--------
	dict:
		A dictionary containing:
		- `num_peaks`: Number of detected peaks in the FFT amplitudes.
		- `peak_amplitude`: Amplitude of the most prominent peak, or None if no peaks.
		- `dominant_frequency`: Frequency corresponding to the most prominent peak, or None if no peaks.
		- `positive_frequencies`: List of positive frequencies.
		- `positive_amplitudes`: List of positive amplitudes.
	"""
	if tokens_fft is None or frequencies is None:
		if verbose:
			console.print("[yellow]FFT results are missing; skipping peak analysis.[/yellow]")
		return {
			"num_peaks": None,
			"peak_amplitude": None,
			"dominant_frequency": None,
			"positive_frequencies": None,
			"positive_amplitudes": None,
		}

	# Extract positive frequencies and amplitudes
	positive_frequencies = frequencies[:len(frequencies) // 2]
	positive_amplitudes = np.abs(tokens_fft[:len(tokens_fft) // 2])

	# Ensure amplitudes are non-zero for meaningful peak detection
	if len(positive_amplitudes) == 0 or np.all(positive_amplitudes == 0):
		if verbose:
			console.print("[yellow]FFT amplitudes are zero or empty; skipping peak analysis.[/yellow]")
		return {
			"num_peaks": 0,
			"peak_amplitude": None,
			"dominant_frequency": None,
			"positive_frequencies": positive_frequencies,
			"positive_amplitudes": positive_amplitudes,
		}

	# Detect peaks in the FFT amplitudes
	try:
		peaks, _ = find_peaks(positive_amplitudes[1:], prominence=min_peak_prominence)
		num_peaks = len(peaks)

		peak_amplitude = (
			np.max(positive_amplitudes[1:][peaks]) if num_peaks > 0 else None
		)
		dominant_frequency = (
			positive_frequencies[peaks[np.argmax(positive_amplitudes[1:][peaks])]]
			if num_peaks > 0 else None
		)

		return {
			"num_peaks": num_peaks,
			"peak_amplitude": peak_amplitude,
			"dominant_frequency": dominant_frequency,
			"positive_frequencies": positive_frequencies.tolist() if positive_frequencies is not None else [],
			"positive_amplitudes": positive_amplitudes.tolist() if positive_amplitudes is not None else [],
		}

	except Exception as e:
		if verbose:
			console.print(f"[red]Error during peak analysis from analyze_fft_peaks function: {e}[/red]")
		return {
			"num_peaks": None,
			"peak_amplitude": None,
			"dominant_frequency": None,
			"positive_frequencies": positive_frequencies.tolist() if positive_frequencies is not None else [],
			"positive_amplitudes": positive_amplitudes.tolist() if positive_amplitudes is not None else [],
		}
	
def calculate_dynamic_cutoff(
	tokens_signal: np.ndarray,
	verbose: bool,
	peak_amplitude: float = 0,
	min_tokens: float = 0,
	percentile_values: list = [5, 10, 15]  # Iterating percentile settings
) -> tuple:
	"""
	Calculate the dynamic cutoff for a signal based on its median, dominant peak amplitude, and lower percentiles.

	Parameters:
	-----------
	tokens_signal : np.ndarray
		The signal to analyze.
	peak_amplitude : float, optional
		Amplitude of the most prominent peak in the signal. Default is 0.
	min_tokens : float, optional
		Minimum observed tokens per page to ensure cutoff is meaningful. Default is 0.
	percentile_values : list, optional
		List of percentiles to test for lower bound enforcement.

	Returns:
	--------
	tuple:
		- float: The calculated dynamic cutoff for the signal.
		- float: The final percentile used for the dynamic cutoff.	
	"""
	if len(tokens_signal) == 0:
		if verbose:
			console.print("[yellow]Signal is empty; returning zero as dynamic cutoff.[/yellow]")
		return 0.0

	best_cutoff = None
	final_percentile = None
	for percentile in percentile_values:
		dynamic_cutoff_signal = max(
			np.median(tokens_signal) - peak_amplitude,
			np.percentile(tokens_signal, percentile)
		)

		# Ensure the cutoff respects the minimum token count
		cutoff_value = max(dynamic_cutoff_signal, min_tokens)

		# Keep the best cutoff (largest non-zero value)
		if best_cutoff is None or cutoff_value > best_cutoff:
			best_cutoff = cutoff_value
			final_percentile = percentile

	return best_cutoff, final_percentile

def detect_relative_peaks(
	tokens_signal: np.ndarray,
	prominence_values: list = [0.05, 0.1, 0.15],  # Iterating prominence thresholds
	distance_factors: list = [10, 20, 30]  # Iterating distance settings as fractions of signal length
) -> dict:
	"""
	Perform relative peak detection on the signal and calculate statistics about the detected peaks.

	Parameters:
	-----------
	tokens_signal : np.ndarray
		The signal to analyze.
	prominence_values : list, optional
		List of prominence thresholds to test. Default is [0.05, 0.1, 0.15] of the signal’s standard deviation.
	distance_factors : list, optional
		List of distance values as fractions of signal length. Default is [10, 20, 30].

	Returns:
	--------
	dict:
		A dictionary containing:
		- `relative_num_peaks`: Number of detected peaks.
		- `avg_prominence`: Average prominence of the detected peaks.
		- `relative_peaks`: Indices of the detected peaks.
		- `relative_prominences`: Flattened prominences of the detected peaks.
		- `relative_left_bases`: Flattened left bases of the detected peaks.
		- `relative_right_bases`: Flattened right bases of the detected peaks.
		- `prominence_threshold`: Prominence threshold used for peak detection.
		- `distance_factor`: Distance factor used for peak detection.
	"""
	if len(tokens_signal) == 0:
		console.print("[yellow]Signal is empty; no peaks detected.[/yellow]")
		return {
			"relative_num_peaks": 0,
			"avg_prominence": None,
			"relative_peaks": [],
			"relative_prominences": [],
			"relative_left_bases": [],
			"relative_right_bases": [],
			"prominence_threshold": None,
			"distance_factor": None
		}

	best_result = None
	max_peaks = 0  # Track the configuration with the highest number of peaks detected
	last_prominence, last_distance_factor = None, None  # Keep track of the last attempted values

	for prominence in prominence_values:
		for distance_factor in distance_factors:
			distance = max(1, len(tokens_signal) // distance_factor)
			last_prominence, last_distance_factor = prominence, distance_factor  # Track last values

			# Detect peaks
			relative_peaks, relative_properties = find_peaks(tokens_signal, prominence=prominence, distance=distance)
			num_peaks = len(relative_peaks)

			# Store prominence properties once instead of multiple dictionary lookups
			prominences = relative_properties.get("prominences", [])
			avg_prominence = np.mean(prominences) if num_peaks > 0 else None

			# Keep the best configuration
			if num_peaks > max_peaks:
				max_peaks = num_peaks
				best_result = {
					"relative_num_peaks": num_peaks,
					"avg_prominence": avg_prominence,
					"relative_peaks": relative_peaks.tolist(),
					"relative_prominences": prominences.tolist(),
					"relative_left_bases": relative_properties.get("left_bases", []).tolist(),
					"relative_right_bases": relative_properties.get("right_bases", []).tolist(),
					"prominence_threshold": prominence,
					"distance_factor": distance_factor
				}

	# If no peaks were found in any iteration, return the last tested configuration
	if not best_result:
		console.print("[yellow]No peaks detected with any prominence or distance factor settings.[/yellow]")
		return {
			"relative_num_peaks": 0,
			"avg_prominence": None,
			"relative_peaks": [],
			"relative_prominences": [],
			"relative_left_bases": [],
			"relative_right_bases": [],
			"prominence_threshold": last_prominence,
			"distance_factor": last_distance_factor
		}

	return best_result

def calculate_autocorrelation(signal: np.ndarray) -> float:
	"""
	Calculate the maximum autocorrelation of a signal.

	Parameters:
	-----------
	signal : np.ndarray
		The input signal.

	Returns:
	--------
	float:
		The maximum autocorrelation value.
	"""
	if len(signal) == 0:
		console.print("[yellow]Signal is empty; autocorrelation is undefined.[/yellow]")
		return 0.0
	autocorr = np.correlate(signal, signal, mode="full")
	return np.max(autocorr[len(autocorr) // 2:])

def calculate_signal_envelope(signal: np.ndarray) -> dict:
	"""
	Calculate the upper and lower envelopes of a signal.

	Parameters:
	-----------
	signal : np.ndarray
		The input signal.

	Returns:
	--------
	dict:
		A dictionary containing the upper and lower envelopes.
	"""
	if len(signal) == 0:
		console.print("[yellow]Signal is empty; envelope is undefined.[/yellow]")
		return {"upper_envelope": 0.0, "lower_envelope": 0.0}

	upper_envelope = np.max(np.abs(signal))
	lower_envelope = -upper_envelope

	return {"upper_envelope": upper_envelope, "lower_envelope": lower_envelope}

def calculate_spectral_features(
	positive_amplitudes: np.ndarray, positive_frequencies: np.ndarray, verbose: bool
) -> dict:
	"""
	Calculate spectral features: magnitude, centroid, and bandwidth.

	Parameters:
	-----------
	positive_amplitudes : np.ndarray
		Positive amplitudes from the FFT.
	positive_frequencies : np.ndarray
		Positive frequencies from the FFT.

	Returns:
	--------
	dict:
		A dictionary containing spectral magnitude, centroid, bandwidth, and max amplitude and frequency.
	"""
	if positive_amplitudes is None or positive_frequencies is None or len(positive_amplitudes) == 0 or len(positive_frequencies) == 0:
		if verbose:
			console.print("[yellow]FFT data is empty or None; spectral features cannot be calculated.[/yellow]")
		return {"spectral_magnitude": 0.0, "spectral_centroid": None, "spectral_bandwidth": None}

	spectral_magnitude = np.sum(positive_amplitudes)
	spectral_centroid = (
		np.sum(positive_frequencies * positive_amplitudes) / spectral_magnitude
		if spectral_magnitude > 0 else None
	)
	spectral_bandwidth = (
		np.sqrt(
			np.sum((positive_frequencies - spectral_centroid) ** 2 * positive_amplitudes)
			/ spectral_magnitude
		)
		if spectral_centroid else None
	)

	return {
		"spectral_magnitude": spectral_magnitude,
		"spectral_centroid": spectral_centroid,
		"spectral_bandwidth": spectral_bandwidth,
		"amplitude_max": np.max(positive_amplitudes) if len(positive_amplitudes) > 0 else None,
		"frequency_max": np.max(positive_frequencies) if len(positive_frequencies) > 0 else None,
	}

def log_metrics(metrics: dict, title: str):
	"""
	Log metrics for debugging purposes.

	Parameters:
	-----------
	metrics : dict
		The metrics to log.
	title : str
		A descriptive title for the metrics.
	"""
	console.print(f"[bright_cyan]{title}[/bright_cyan]")
	for key, value in metrics.items():
		console.print(f"{key}: {value}")

def calculate_signal_metrics(
	tokens_signal: np.ndarray,
	use_signal_type: str,
	min_tokens: float,
	prominence: float = None,
	distance: int = None,
	verbose: bool = True,
) -> dict:
	"""
	Calculate comprehensive metrics for a signal.

	Parameters:
	-----------
	tokens_signal : np.ndarray
		The signal to analyze.
	use_signal_type : str
		The type of signal being analyzed.
	min_tokens : float
		Minimum number of tokens required for meaningful analysis.
	prominence : float, optional
		Minimum prominence of peaks for peak detection. Default is None.
	distance : int, optional
		Minimum distance between peaks for peak detection. Default is None.
	verbose : bool, optional
		Whether to display verbose output. Default is True.

	Returns:
	--------
	dict:
		A dictionary containing the calculated metrics.
	"""
	if tokens_signal is None or len(tokens_signal) == 0:
		console.print(f"[bright_red]Error: Empty or invalid signal for {use_signal_type}.[/bright_red]")
		return {}

	# try:
	# FFT Analysis
	positive_amplitudes, positive_frequencies, snr_threshold, stationarity_threshold = calculate_fft(tokens_signal, verbose)
	fft_metrics = analyze_fft_peaks(
		positive_amplitudes, positive_frequencies, verbose, min_peak_prominence=prominence or 0.01
	)

	fft_results = {
		"dominant_frequency": fft_metrics.get("dominant_frequency"),
		"amplitude_max": fft_metrics.get("peak_amplitude"),
		"num_fft_peaks": fft_metrics.get("num_peaks"),
		"positive_frequencies": positive_frequencies.tolist() if positive_frequencies is not None else [],
		"positive_amplitudes": positive_amplitudes.tolist() if positive_amplitudes is not None else [],
		"snr_threshold": snr_threshold,
		"stationarity_threshold": stationarity_threshold,
	}

	# Dynamic Cutoff
	dynamic_cutoff, cutoff_percentile = calculate_dynamic_cutoff(
		tokens_signal=tokens_signal,
		verbose=verbose,
		peak_amplitude=fft_metrics.get("peak_amplitude", 0),
		min_tokens=min_tokens,
	)

	# Relative Peaks
	peak_metrics = detect_relative_peaks(
		tokens_signal=tokens_signal,
		prominence_values=[0.05, 0.1, 0.15],  # Updated to list format for iteration
		distance_factors=[10, 20, 30]  # Updated to list format for iteration
	)

	# Ensure safe handling of missing or empty peak data
	peak_results = {
		"relative_num_peaks": peak_metrics.get("relative_num_peaks", 0),
		"avg_prominence": np.mean(peak_metrics["relative_prominences"]) if peak_metrics.get("relative_prominences") else None,
		"prominence_min": np.min(peak_metrics["relative_prominences"]) if peak_metrics.get("relative_prominences") else None,
		"prominence_max": np.max(peak_metrics["relative_prominences"]) if peak_metrics.get("relative_prominences") else None,
		"relative_peaks": peak_metrics.get("relative_peaks", []),
		"relative_prominences": peak_metrics.get("relative_prominences", []),
		"relative_left_bases": peak_metrics.get("relative_left_bases", []),
		"relative_right_bases": peak_metrics.get("relative_right_bases", []),
		"prominence_threshold": peak_metrics.get("prominence_threshold"),  # Track best prominence setting
		"distance_factor": peak_metrics.get("distance_factor")  # Track best distance factor
	}

	# Autocorrelation
	max_autocorr = calculate_autocorrelation(tokens_signal)
	autocorr_results = {"max_autocorrelation": max_autocorr}

	# Signal Envelope
	envelope_metrics = calculate_signal_envelope(tokens_signal)

	# Spectral Features
	spectral_features = calculate_spectral_features(
		positive_amplitudes=positive_amplitudes, positive_frequencies=positive_frequencies,
		verbose=verbose
	)

	spectral_results = {
		"spectral_magnitude": spectral_features.get("spectral_magnitude", 0.0),
		"spectral_centroid": spectral_features.get("spectral_centroid"),
		"spectral_bandwidth": spectral_features.get("spectral_bandwidth"),
		"frequency_max": spectral_features.get("frequency_max"),
	}

	# Compile All Results
	metrics = {
		"signal_type": use_signal_type,
		"dynamic_cutoff": dynamic_cutoff,
		"dynamic_cutoff_percentile": cutoff_percentile,
		**fft_results,
		**peak_results,
		**autocorr_results,
		**envelope_metrics,
		**spectral_results,
	}

	# Logging
	if verbose:
		log_metrics(metrics, f"Metrics for {use_signal_type}")

	return metrics

	# except Exception as e:
	#     console.print(f"[bright_red]Error calculating metrics for {use_signal_type}: {e}[/bright_red]")
	#     return {}