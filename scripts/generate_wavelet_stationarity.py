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

def check_wavelet_stationarity(
	signal: np.ndarray, 
	signal_type: str, 
	max_lag_range: list = [5, 10, 15],  
	significance_level: float = 0.05
) -> dict:
	"""
	Check the stationarity of a signal using the Augmented Dickey-Fuller and Kwiatkowski-Phillips-Schmidt-Shin tests.

	Combined Result Interpretation:
	- ADF p-value ≤ significance and KPSS p-value > significance: Signal is stationary.
	- ADF p-value > significance and KPSS p-value ≤ significance: Signal is non-stationary.
	- Both tests significant (p-value ≤ significance): Potential trend-stationary; requires further inspection.
	- Both tests non-significant (p-value > significance): Likely stationary but may require confirmation.

	Parameters:
	----------
	signal : np.ndarray
		The signal to check for stationarity.
	signal_type : str
		The type of signal being analyzed (e.g., raw or smoothed).
	max_lag_range : list, optional
		List of max_lag values to test (default: [5, 10, 15]).
	significance_level : float, optional
		The significance level for the tests (default: 0.05).

	Returns:
	--------
	dict:
		- is_stationary (bool): Whether the signal is stationary.
		- best_max_lag (int or None): Best max_lag that yielded stationarity.
		- ADF p-value (float): p-value from the ADF test.
		- KPSS p-value (float or None): p-value from the KPSS test.
		- ADF statistic (float): Test statistic from the ADF test.
		- KPSS statistic (float or None): Test statistic from the KPSS test.
		- interpretation (str): Explanation of the result.
	"""
	final_interpretation = None
	for max_lag in max_lag_range:
		console.print(f"[blue]Testing stationarity with max_lag={max_lag}...[/blue]")

		# Augmented Dickey-Fuller Test
		adf_stat, adf_pvalue, _, _, _, _ = adfuller(signal, maxlag=max_lag)
		console.print(f"[violet]ADF Test for {signal_type}: Statistic={adf_stat:.4f}, p-value={adf_pvalue:.4f}[/violet]")

		# Kwiatkowski-Phillips-Schmidt-Shin Test
		try:
			kpss_stat, kpss_pvalue, _, _ = kpss(signal, regression='c')
			console.print(f"[violet]KPSS Test for {signal_type}: Statistic={kpss_stat:.4f}, p-value={kpss_pvalue:.4f}[/violet]")
		except ValueError as e:
			console.print(f"[bright_red]Error in KPSS test: {e}[/bright_red]")
			kpss_stat, kpss_pvalue = None, None  # Handle KPSS failure gracefully

		# --- Interpretation of Stationarity ---
		if adf_pvalue <= significance_level and (kpss_pvalue is None or kpss_pvalue > significance_level):
			interpretation = f"Stationary at max_lag={max_lag}. ADF rejected unit root (p={adf_pvalue:.4f}), KPSS failed to reject stationarity (p={kpss_pvalue})."
			console.print(f"[green]{interpretation}[/green]")
			return {
				"is_stationary": True,
				"best_max_lag": max_lag,
				"ADF p-value": adf_pvalue,
				"KPSS p-value": kpss_pvalue,
				"ADF statistic": adf_stat,
				"KPSS statistic": kpss_stat,
				"interpretation": interpretation
			}

		elif adf_pvalue > significance_level and (kpss_pvalue is not None and kpss_pvalue <= significance_level):
			interpretation = f"Non-stationary at max_lag={max_lag}. ADF failed to reject unit root (p={adf_pvalue:.4f}), KPSS detected trend-stationarity (p={kpss_pvalue:.4f})."
			console.print(f"[red]{interpretation}[/red]")
			final_interpretation = interpretation

		elif adf_pvalue <= significance_level and (kpss_pvalue is not None and kpss_pvalue <= significance_level):
			interpretation = f"Conflicting results at max_lag={max_lag}. ADF suggests stationarity (p={adf_pvalue:.4f}), but KPSS indicates trend-stationarity (p={kpss_pvalue:.4f}). Further preprocessing may be needed."
			console.print(f"[yellow]{interpretation}[/yellow]")
			final_interpretation = interpretation

		else:
			interpretation = f"Likely stationary at max_lag={max_lag}, but weak statistical evidence. Both tests fail to provide strong conclusions (ADF p={adf_pvalue:.4f}, KPSS p={kpss_pvalue})."
			console.print(f"[green]{interpretation}[/green]")
			return {
				"is_stationary": True,
				"best_max_lag": max_lag,
				"ADF p-value": adf_pvalue,
				"KPSS p-value": kpss_pvalue,
				"ADF statistic": adf_stat,
				"KPSS statistic": kpss_stat,
				"interpretation": interpretation
			}

	# If no stationary result was found, return the last tested result
	console.print("[red]Signal remains non-stationary for all max_lag values tested.[/red]")
	return {
		"is_stationary": False,
		"best_max_lag": None,
		"ADF p-value": adf_pvalue,
		"KPSS p-value": kpss_pvalue,
		"ADF statistic": adf_stat,
		"KPSS statistic": kpss_stat,
		"interpretation": f"Signal remains non-stationary despite all tested max_lag values." if final_interpretation is None else final_interpretation + " (best result from all max_lag values)"
	}

def preprocess_signal_for_stationarity(signal: np.ndarray, signal_type: str, max_lag: int = 10, significance_level: float = 0.05) -> tuple:
	"""
	Preprocess a signal to achieve stationarity by applying detrending or differencing if necessary. The function first checks the stationarity of the input signal using the Augmented Dickey-Fuller and Kwiatkowski-Phillips-Schmidt-Shin tests. If the signal is non-stationary, it applies detrending and differencing sequentially until the signal becomes stationary. We also iterate through the following transformations:
	- Original signal
	- Linear detrending
	- Constant detrending
	- First-order differencing
	- Second-order differencing

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
	# Initialize stationarity_result to avoid potential NameError
	stationarity_result = {
		"is_stationary": False,
		"best_max_lag": None,
		"ADF p-value": None,
		"KPSS p-value": None,
		"ADF statistic": None,
		"KPSS statistic": None,
		"transformation": "None (original signal used)"
	}

	# List of transformations to try in sequence
	transformations = [
		("Original", signal),
		("Linear Detrending", apply_detrending(signal, method="linear")),
		("Constant Detrending", apply_detrending(signal, method="constant")),
		("First-Order Differencing", apply_differencing(signal, order=1)),
		("Second-Order Differencing", apply_differencing(signal, order=2))
	]

	for method, transformed_signal in transformations:
		if transformed_signal is None:
			continue

		console.print(f"[blue]Testing stationarity with {method}...[/blue]")
		stationarity_result = check_wavelet_stationarity(transformed_signal, signal_type, max_lag, significance_level)
		stationarity_result["transformation"] = method

		if stationarity_result["is_stationary"]:
			console.print(f"[green]Signal is stationary after {method}.[/green]")
			return transformed_signal, stationarity_result

	console.print("[red]Signal remains non-stationary despite all preprocessing attempts.[/red]")
	return signal, stationarity_result  # Return original signal with last stationarity check