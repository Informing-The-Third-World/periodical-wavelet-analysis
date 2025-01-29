# Standard library imports
import warnings

# Third-party imports
import numpy as np
from rich.console import Console
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import detrend

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

## WAVELET STATIONARITY FUNCTIONS
def apply_differencing(signal: np.ndarray, order: int = 1) -> np.ndarray:
	"""
	Apply differencing to a signal to remove trends and achieve stationarity. If the order is less than 1, the function returns None with a warning. Differencing is only used when a signal is non-stationary.

	Parameters:
	-----------
	signal : np.ndarray
		The input signal.
	order : int, optional
		The order of differencing. Default is 1.

	Returns:
	--------
	np.ndarray or None:
		The differenced signal if successful; None if the order is invalid.
	"""
	if order < 1:
		console.print("[red]Order of differencing must be at least 1. Returning None.[/red]")
		return None

	try:
		differenced_signal = np.diff(signal, n=order)
		return differenced_signal
	except Exception as e:
		console.print(f"[red]Error applying differencing: {e}. Returning None.[/red]")
		return None

def apply_detrending(signal: np.ndarray, method: str = "linear") -> np.ndarray:
	"""
	Remove trends from a signal using linear or polynomial detrending. If the method is invalid, the function returns None with a warning. Detrending is only used when a signal is non-stationary.

	Parameters:
	-----------
	signal : np.ndarray
		The input signal.
	method : str, optional
		The detrending method. Options are "linear" (default) or "constant".
		- "linear": Removes a linear trend.
		- "constant": Removes the mean of the signal.

	Returns:
	--------
	np.ndarray or None:
		The detrended signal if successful; None if the method is invalid.
	"""
	if method not in ["linear", "constant"]:
		console.print("[red]Invalid method. Use 'linear' or 'constant'. Returning None.[/red]")
		return None

	try:
		return detrend(signal, type=method)
	except Exception as e:
		console.print(f"[red]Error applying detrending: {e}. Returning None.[/red]")
		return None

def check_wavelet_stationarity(signal: np.ndarray, signal_type: str, max_lag: int = 10, significance_level: float = 0.05) -> dict:
	"""
	Check the stationarity of a signal using the Augmented Dickey-Fuller and Kwiatkowski-Phillips-Schmidt-Shin tests.

	Combined Result Interpretation:
	- ADF p-value ≤ significance and KPSS p-value > significance: Signal is stationary.
	- ADF p-value > significance and KPSS p-value ≤ significance: Signal is non-stationary.
	- Both tests significant (p-value ≤ significance): Potential trend-stationary; requires further inspection.
	- Both tests non-significant (p-value > significance): Likely stationary but may require confirmation.

	Parameters:
	-----------
	signal : np.ndarray
		The signal to check for stationarity.
	signal_type : str
		The type of signal being analyzed (e.g., raw or smoothed).
	max_lag : int, optional
		The maximum lag to consider in the ADF test. Default is 10.
	significance_level : float, optional
		The significance level for the tests. Default is 0.05.

	Returns:
	--------
	dict:
		- is_stationary: bool, whether the signal is stationary.
		- ADF p-value: float, p-value from the ADF test.
		- KPSS p-value: float, p-value from the KPSS test.
		- ADF statistic: float, test statistic from the ADF test.
		- KPSS statistic: float, test statistic from the KPSS test.
	"""
	# Augmented Dickey-Fuller Test
	adf_stat, adf_pvalue, _, _, _, _ = adfuller(signal, maxlag=max_lag)
	console.print(f"[violet]ADF Test for {signal_type}: Statistic={adf_stat:.4f}, p-value={adf_pvalue:.4f}[/violet]")

	# Kwiatkowski-Phillips-Schmidt-Shin Test
	try:
		kpss_stat, kpss_pvalue, _, _ = kpss(signal, regression='c')
		console.print(f"[violet]KPSS Test for {signal_type}: Statistic={kpss_stat:.4f}, p-value={kpss_pvalue:.4f}[/violet]")
	except ValueError as e:
		console.print(f"[bright_red]Error in KPSS test: {e}[/bright_red]")
		return {
			"is_stationary": False,
			"ADF p-value": adf_pvalue,
			"KPSS p-value": None,
			"ADF statistic": adf_stat,
			"KPSS statistic": None
		}

	# Combined Result Interpretation
	if adf_pvalue <= significance_level and kpss_pvalue > significance_level:
		is_stationary = True
		console.print("[green]Signal is stationary.[/green]")
	elif adf_pvalue > significance_level and kpss_pvalue <= significance_level:
		is_stationary = False
		console.print("[red]Signal is non-stationary.[/red]")
	elif adf_pvalue <= significance_level and kpss_pvalue <= significance_level:
		console.print("[yellow]Conflicting results: Further inspection needed.[/yellow]")
		is_stationary = False
	else:
		is_stationary = True
		console.print("[green]Likely stationary but requires confirmation.[/green]")

	return {
		"is_stationary": is_stationary,
		"ADF p-value": adf_pvalue,
		"KPSS p-value": kpss_pvalue,
		"ADF statistic": adf_stat,
		"KPSS statistic": kpss_stat
	}

def preprocess_signal_for_stationarity(signal: np.ndarray, signal_type: str, max_lag: int = 10, significance_level: float = 0.05) -> tuple:
	"""
	Preprocess a signal to achieve stationarity by applying detrending or differencing if necessary. The function first checks the stationarity of the input signal using the Augmented Dickey-Fuller and Kwiatkowski-Phillips-Schmidt-Shin tests. If the signal is non-stationary, it applies detrending and differencing sequentially until the signal becomes stationary.

	A signal of token frequency might be non-stationary if it exhibits trends or seasonality, which can affect the accuracy of wavelet analysis. Preprocessing the signal for stationarity is essential for reliable wavelet decomposition and feature extraction.

	Parameters:
	-----------
	signal : np.ndarray
		The input signal.
	signal_type : str
		The type of signal being analyzed (e.g., "raw", "smoothed").
	max_lag : int, optional
		Maximum lag for the ADF test.
	significance_level : float, optional
		Significance level for stationarity tests.

	Returns:
	--------
	tuple:
		- processed_signal (np.ndarray): The processed signal (stationary if preprocessing is successful).
		- stationarity_results (dict): Results of the stationarity tests.
	"""
	stationarity_result = check_wavelet_stationarity(signal, signal_type, max_lag, significance_level)
	processed_signal = signal  # Start with the original signal

	if stationarity_result["is_stationary"]:
		console.print("[bright_green]Signal is already stationary. No preprocessing needed.[/bright_green]")
		return processed_signal, stationarity_result
	
	console.print("[yellow]Signal is not stationary. Applying detrending...[/yellow]")
	detrended_signal = apply_detrending(signal, method="linear")
	
	# Re-check stationarity after detrending
	stationarity_result = check_wavelet_stationarity(detrended_signal, signal_type, max_lag, significance_level)
	if stationarity_result["is_stationary"]:
		console.print("[bright_green]Signal is stationary after detrending.[/bright_green]")
		return detrended_signal, stationarity_result
	
	console.print("[yellow]Signal is still not stationary. Applying first-order differencing...[/yellow]")
	differenced_signal = apply_differencing(detrended_signal, order=1)
	
	# Final stationarity check
	stationarity_result = check_wavelet_stationarity(differenced_signal, signal_type, max_lag, significance_level)
	if stationarity_result["is_stationary"]:
		console.print("[bright_green]Signal is stationary after differencing.[/bright_green]")
		return differenced_signal, stationarity_result
	else:
		console.print("[red]Signal remains non-stationary despite preprocessing.[/red]")
		return differenced_signal, stationarity_result
