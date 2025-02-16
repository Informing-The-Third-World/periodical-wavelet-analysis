# Standard library imports
import warnings
from typing import Any, Tuple, Union

# Third-party imports
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from rich.console import Console
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.stats import wasserstein_distance
from fastdtw import fastdtw
from minineedle import needle, core
from tqdm import tqdm

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

## WEIGHTS & CONFIGURATION FOR RANKING

SIGNAL_METRIC_WEIGHTS = {
	# **Cluster 1: Prominence & Amplitude-Based Metrics (Moderate Priority)**
	"emd_value_normalized": 0.3,
	"smoothness_normalized": 0.3,

	# **Cluster 2: Spectral & Structural Fidelity (Highest Priority)**
	"wavelet_mse_normalized": 0.6,
	"wavelet_psnr_normalized": 0.6,
	"wavelet_entropy_normalized": 0.5,
	"wavelet_energy_entropy_normalized": 0.5,
	"wavelet_sparsity_normalized": 0.4,
	"correlation_normalized": 0.4,
	"avg_variance_across_levels_normalized": 0.4,

	# **Cluster 3: Structural Fidelity (Moderate Priority)**
	"kl_divergence_normalized": 0.4,  # Ensured consistency with reconstruction
}

RECONSTRUCTION_METRIC_WEIGHTS = {
	# **Cluster 1: Prominence & Amplitude-Based Metrics (Moderate Importance)**
	"prominence_min_diff_normalized": 0.3,
	"positive_amplitudes_dtw_normalized": 0.35,
	"positive_amplitudes_euclidean_normalized": 0.35,
	"positive_amplitudes_wasserstein_normalized": 0.35,
	"avg_prominence_diff_normalized": 0.3,
	"relative_prominences_avg_diff_normalized": 0.3,
	"relative_prominences_wasserstein_normalized": 0.3,
	"positive_frequencies_dtw_normalized": 0.35,
	"positive_frequencies_euclidean_normalized": 0.35,
	"positive_frequencies_wasserstein_normalized": 0.35,

	# **Cluster 2: Spectral & Structural Fidelity (Highest Importance)**
	"spectral_centroid_diff_normalized": 0.6,
	"spectral_magnitude_diff_normalized": 0.6,
	"dynamic_cutoff_diff_normalized": 0.5,
	"relative_prominences_total_diff_normalized": 0.5,
	"prominence_max_diff_normalized": 0.5,
	"amplitude_max_diff_normalized": 0.5,
	"frequency_max_diff_normalized": 0.5,
	"spectral_bandwidth_diff_normalized": 0.5,
	"num_fft_peaks_diff_normalized": 0.4,
	"relative_num_peaks_diff_normalized": 0.4,  # Fixed typo
	"dominant_frequency_diff_normalized": 0.5,
	"max_autocorrelation_diff_normalized": 0.5,  # Ensured it's here

	# **Cluster 3: Alignment-Based Metrics (Lower Importance)**
	"relative_right_bases_global_alignment_score_normalized": 0.1,
	"relative_right_bases_array_alignment_score_normalized": 0.1,
	"relative_left_bases_global_alignment_score_normalized": 0.1,
	"relative_left_bases_array_alignment_score_normalized": 0.1,
	"upper_envelope_diff_normalized": 0.2,
	"lower_envelope_diff_normalized": 0.2,
	"relative_peaks_array_alignment_score_normalized": 0.2,
	"relative_peaks_global_alignment_score_normalized": 0.2,
}

LOWER_IS_BETTER_METRICS = [
	# Prominence & amplitude diffs
	"prominence_min_diff_normalized",
	"prominence_max_diff_normalized",
	"avg_prominence_diff_normalized",
	"relative_prominences_avg_diff_normalized",
	"relative_prominences_total_diff_normalized",
	"positive_amplitudes_dtw_normalized",
	"positive_amplitudes_euclidean_normalized",
	"positive_amplitudes_wasserstein_normalized",
	"positive_frequencies_dtw_normalized",
	"positive_frequencies_euclidean_normalized",
	"positive_frequencies_wasserstein_normalized",

	# Spectral & structural fidelity
	"dynamic_cutoff_diff_normalized",
	"frequency_max_diff_normalized",
	"spectral_bandwidth_diff_normalized",
	"num_fft_peaks_diff_normalized",
	"relative_num_peaks_diff_normalized",
	"dominant_frequency_diff_normalized",
	"spectral_magnitude_diff_normalized",
	"spectral_centroid_diff_normalized",

	# Global vs. array alignment scores (lower distance = better)
	"relative_right_bases_global_alignment_score_normalized",
	"relative_right_bases_array_alignment_score_normalized",
	"relative_left_bases_global_alignment_score_normalized",
	"relative_left_bases_array_alignment_score_normalized",
	"relative_peaks_array_alignment_score_normalized",
	"relative_peaks_global_alignment_score_normalized",

	# Envelopes, correlation, etc.
	"upper_envelope_diff_normalized",
	"lower_envelope_diff_normalized",
	"kl_divergence_normalized",     
	"emd_value_normalized",        
	"max_autocorrelation_diff_normalized"
]

def generate_metric_config(metric: str, weight: float) -> dict:
	"""
	Generates a metric configuration dictionary.

	Parameters:
	-----------
	metric : str
		The metric name.
	weight : float
		The weight of the metric.

	Returns:
	--------
	metric_config : dict
		Dictionary containing the metric configuration.
	"""
	metric = metric.split("_normalized")[0]  # Remove "_normalized" suffix
	return {
		"metric": metric,
		"original_weight": weight,
		"normalized_weight": None,
		"was_normalized": False,
		"was_inverted": False,
		"was_shared": False,
		"was_specific": False,
		"max_value": None,
		"min_value": None,
		"std_dev": None,
		"mean": None,
		"variance": None,
		"presence": None,
		"ignore_metric": False,
		"removal_reason": None,
		"was_log_transformed": False,
		"low_variance_detected": False,
	}

def generate_ranking_config(signal_type: str, prefix: str = "") -> dict:
	"""
	Generates a ranking configuration dictionary based on the provided signal and reconstruction metric weights. It logs the initial weights, transformations, and exclusions of metrics for future reference. The configuration is used to track the normalization, inversion, and correlation of metrics during the ranking process. 

	Parameters:
	-----------
	signal_type : str
		The type of signal being analyzed.
	prefix : str
		Prefix for column names (e.g., "combined_" for merged datasets).
	
	Returns:
	--------
	ranking_config : dict
		Dictionary containing the ranking configuration.
	"""
	ranking_config = {
		"signal_type": signal_type,
		"metrics": []
	}
	if prefix:
		ranking_config["comparison_prefix"] = prefix
	for metric, weight in SIGNAL_METRIC_WEIGHTS.items():
		ranking_config["metrics"].append(generate_metric_config(metric, weight))
	
	for metric, weight in RECONSTRUCTION_METRIC_WEIGHTS.items():
		ranking_config["metrics"].append(generate_metric_config(metric, weight))
	console.print(f"Generated ranking configuration for '{signal_type}' signal with {len(ranking_config['metrics'])} metrics.", style="bright_cyan")
	return ranking_config

## WAVELET RANKING PREPROCESSING FUNCTIONS
def run_global_sequence_alignment(
	original_sequence: list, 
	reconstructed_sequence: list, 
	placeholder: int = -999999
) -> int:
	"""
	Applies global sequence alignment on the implied zero/missing values 
	within a reconstructed_sequence using minineedle, with placeholders.

	Parameters
	----------
	original_sequence : list
		The target sequence to be used in the alignment.
	reconstructed_sequence : list
		The reconstructed_sequence containing the observed sequence.
	placeholder : int
		The placeholder value to be used in the alignment.
		
	Returns
	-------
	alignment_score : int
		The alignment score. If the alignment fails, returns 0.
		A higher score indicates a better alignment.
	"""
	# Convert NaNs to the placeholder
	observed_sequence = [
		float(p) if pd.notna(p) else placeholder 
		for p in reconstructed_sequence
	]
	
	# Check if all are placeholders
	if all(val == placeholder for val in observed_sequence):
		return 0
	
	# Create Needleman-Wunsch global alignment instance
	alignment = needle.NeedlemanWunsch(observed_sequence, original_sequence)
	# Adjust scoring matrix if desired
	alignment.change_matrix(core.ScoreMatrix(match=4, miss=-0.5, gap=-1))

	try:
		alignment.align()
		alignment_score = alignment.get_score()
		return alignment_score

	except Exception as e:
		console.print(f"[red]Global sequence alignment error: {e}[/red]")
		return 0

def align_arrays(original_array: Union[list, np.ndarray], 
	reconstructed_array: Union[list, np.ndarray], 
	threshold: float = 1e-6
) -> float:
	"""
	Example of a numeric alignment measure (simple approach).
	Returns a fraction of element-wise matches (within a threshold)
	for arrays of equal length.
	
	If arrays differ in length, returns 0.
	
	A higher score indicates a closer numeric match (1.0 = perfect).
	"""
	# Convert to numpy
	orig_np = np.array(original_array)
	recon_np = np.array(reconstructed_array)

	if len(orig_np) != len(recon_np) or len(orig_np) == 0:
		return 0.0
	
	diff = np.abs(orig_np - recon_np)
	matches = diff < threshold
	return np.sum(matches) / len(orig_np)

def compare_distributions(
	original_array: Union[list, np.ndarray], 
	reconstructed_array: Union[list, np.ndarray], 
	method: str = "dtw"
) -> float:
	"""
	Compare two distributions or sequences using the specified method.

	Parameters:
	----------
	original_array : array-like
		First array for comparison.
	reconstructed_array : array-like
		Second array for comparison.
	method : str, optional
		Method to use for comparison. Options:
		- "dtw": Dynamic Time Warping for alignment-based comparison.
		- "euclidean": Simple Euclidean distance for aligned arrays.
		- "wasserstein": Wasserstein (Earth Mover's) distance for distributions.

	Returns:
	-------
	float
		Distance or similarity score between the two arrays.
	"""
	# Convert to numpy arrays if necessary
	orig_np = np.array(original_array, dtype=float)
	recon_np = np.array(reconstructed_array, dtype=float)

	try:
		if method == "dtw":
			distance, _ = fastdtw(orig_np, recon_np)
			return distance
		elif method == "euclidean":
			if len(orig_np) != len(recon_np):
				raise ValueError("Euclidean distance requires arrays of equal length.")
			return np.sqrt(np.sum((orig_np - recon_np) ** 2))
		elif method == "wasserstein":
			return wasserstein_distance(orig_np, recon_np)
		else:
			raise ValueError(f"Unknown method: {method}")
	except Exception as e:
		console.print(f"[red]Error comparing distributions using method '{method}': {e}[/red]")
		return np.nan

def compare_prominences_distribution(
	original_array: Union[list, np.ndarray], 
	reconstructed_array: Union[list, np.ndarray]
) -> tuple:
	"""
	Compares prominence distributions using average, total, and Wasserstein distance. Lower values indicate better alignment for average and total differences, and Wasserstein distance.

	Parameters:
	----------
	original_array : list
		The original array.
	reconstructed_array : list
		The reconstructed array.

	Returns:
	-------
	tuple
		A tuple containing the average difference, total difference, and Wasserstein distance.
	"""
	orig_np = np.array(original_array, dtype=float)
	recon_np = np.array(reconstructed_array, dtype=float)

	# Handle edge cases
	if len(orig_np) == 0 or len(recon_np) == 0:
		return np.nan, np.nan, np.nan

	avg_diff = np.abs(np.mean(orig_np) - np.mean(recon_np))
	total_diff = np.abs(np.sum(orig_np) - np.sum(recon_np))
	wasserstein = compare_distributions(orig_np, recon_np, method="wasserstein")
	return avg_diff, total_diff, wasserstein

def process_array(original_value: Any, reconstructed_value: Any) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
	"""
	Convert stringified or list-like arrays to numpy arrays.

	Parameters:
	----------
	original_value : Any
		The original value which can be a stringified array, list-like, or any other type.
	reconstructed_value : Any
		The reconstructed value which can be a stringified array, list-like, or any other type.

	Returns:
	-------
	Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]
		A tuple containing the converted numpy arrays. If an error occurs, returns (None, None).
	"""
	try:
		# Process original_value
		if isinstance(original_value, str) and original_value.startswith("["):
			original_array = np.array(ast.literal_eval(original_value))
		else:
			original_array = np.array(original_value)
		
		# Process reconstructed_value
		if isinstance(reconstructed_value, str) and reconstructed_value.startswith("["):
			reconstructed_array = np.array(ast.literal_eval(reconstructed_value))
		else:
			reconstructed_array = np.array(reconstructed_value)
		
		return original_array, reconstructed_array
	except Exception as e:
		console.print(f"[red]Error processing arrays: {e}[/red]")
		return None, None
	
def convert_to_serializable(value: Any) -> Any:
	"""
	Converts NumPy types to Python native types for JSON serialization.

	Parameters:
	----------
	value : Any
		The value to convert.

	Returns:
	-------
	Any
		The converted value.
	"""
	if isinstance(value, (np.int64, np.int32)):
		return int(value)
	elif isinstance(value, (np.float64, np.float32, np.float16)):
		return float(value)
	elif isinstance(value, np.ndarray):
		return value.tolist()
	else:
		return value

def preprocess_reconstructed_metrics(original_df: pd.DataFrame, reconstructed_df: pd.DataFrame
) -> pd.DataFrame:
	"""
	Compares the outputs from the calculate_signal_metrics function for the original signal versus all reconstructed wavelets. It assumes that the comparison will be within signal type (so not across raw and smoothed results). It also assumes that the original_df contains only one row of data. The function compares the metrics for each wavelet and returns a DataFrame with the comparison results. It uses a combination of sequence alignment, DTW, and other distance measures to compare the metrics. 

	Parameters:
	-----------
	original_df : pd.DataFrame
		DataFrame containing the original signal metrics.
	reconstructed_df : pd.DataFrame
		DataFrame containing the reconstructed signal metrics.
	Returns:
	--------
	reconstruction_df : pd.DataFrame
		DataFrame containing the comparison results and reconstruction scores.
	"""

	if original_df.empty or reconstructed_df.empty:
		console.print("[yellow]One or both DataFrames are empty. Returning empty results.[/yellow]")
		return pd.DataFrame()

	if len(original_df) != 1:
		raise ValueError("original_df must contain exactly one row for comparison.")

	# Identify metrics to compare
	metrics = set(original_df.columns).intersection(set(reconstructed_df.columns)) - {"signal_type"}

	# Extract that single row from original_df as a Series (for easy reference)
	original_series = original_df.iloc[0]

	def compare_metrics_in_row(reconstructed_row: pd.Series) -> pd.Series:
		"""
		Applies the comparison logic to a single row of reconstructed_df.
		Modifies the row in-place, adding new columns with comparison scores.
		"""
		for metric in metrics:
			if metric in original_series and metric in reconstructed_row:
				original_val = original_series[metric]
				recon_val = reconstructed_row[metric]

				# Check if metric is list-like or stringified array
				if isinstance(original_val, (list, np.ndarray)) or (
					isinstance(original_val, str) and original_val.startswith("[")
				):
					original_array, reconstructed_array = process_array(original_val, recon_val)
					
					if original_array is None or reconstructed_array is None or np.isnan(reconstructed_array).all() or np.isnan(original_array).all():
						# Skip if there's an error or invalid arrays
						continue

					if metric in {"relative_peaks", "relative_left_bases", "relative_right_bases"}:
						reconstructed_row[f"{metric}_array_alignment_score"] = align_arrays(
							original_array, reconstructed_array
						)
						reconstructed_row[f"{metric}_global_alignment_score"] = run_global_sequence_alignment(
							original_array, reconstructed_array
						)
					
					elif metric in {"positive_frequencies", "positive_amplitudes"}:
						reconstructed_row[f"{metric}_dtw"] = compare_distributions(
							original_array, reconstructed_array, method="dtw"
						)
						if len(original_array) == len(reconstructed_array):
							reconstructed_row[f"{metric}_euclidean"] = compare_distributions(
								original_array, reconstructed_array, method="euclidean"
							)
						reconstructed_row[f"{metric}_wasserstein"] = compare_distributions(
							original_array, reconstructed_array, method="wasserstein"
						)
					
					elif metric == "relative_prominences":
						avg_diff, total_diff, wasser_dist = compare_prominences_distribution(
							original_array, reconstructed_array
						)
						reconstructed_row[f"{metric}_avg_diff"] = avg_diff
						reconstructed_row[f"{metric}_total_diff"] = total_diff
						reconstructed_row[f"{metric}_wasserstein"] = wasser_dist

				else:
					# Metric is scalar-like, so just compute absolute difference
					try:
						reconstructed_row[f"{metric}_diff"] = abs(
							float(original_val) - float(recon_val)
						)
					except Exception as e:
						reconstructed_row[f"{metric}_diff"] = np.nan

		# Return the modified row
		return reconstructed_row

	# Apply the comparison to each row in reconstructed_df
	# axis=1 ensures we process rows individually
	reconstructed_df = reconstructed_df.apply(compare_metrics_in_row, axis=1)

	return reconstructed_df

def preprocess_signal_metrics(results_df: pd.DataFrame, ranking_config: dict,  should_ignore_low_variance: bool, epsilon_threshold: float = 1e-6, verbose: bool = False) -> tuple:
	"""
	Preprocesses the raw signal metrics DataFrame by handling zero or near-zero variance metrics, log-transforming extreme values in `wavelet_energy_entropy`, and handling complex-valued metrics. It dynamically adjusts weights for low-variance metrics and logs the changes in the ranking configuration. It also log-transforms `wavelet_energy_entropy` if negative values are present. The function then returns the cleaned DataFrame and updated ranking configuration.

	Parameters:
	----------
	results_df : pd.DataFrame
		DataFrame containing raw signal metrics.
	ranking_config : dict
		Dictionary containing the ranking configuration.
	should_ignore_low_variance : bool
		Flag to ignore low-variance metrics.
	epsilon_threshold : float, optional
		Threshold for zero or near-zero variance metrics (default 1e-6).

	Returns:
	--------
	tuple: pd.DataFrame, dict
		results_df : pd.DataFrame
			Cleaned DataFrame with preprocessed metrics.
		ranking_config : dict
			Updated ranking configuration with weight adjustments.
	"""

	# Dynamically handle zero or near-zero variance metrics
	for metric_config in ranking_config["metrics"]:
		metric = metric_config["metric"]
		if metric in results_df.columns:
			std_dev = results_df[metric].std()
			avg_value = results_df[metric].mean()
			
			# Adjust weights instead of outright removal
			if std_dev <= epsilon_threshold:
				if verbose:
					console.print(
						f"[yellow]Low variance for '{metric}' (std: {std_dev:.6f}, mean: {avg_value:.6f}).[/yellow]"
					)
				metric_config["low_variance_detected"] = True
				# signal_metric_weights[metric] *= 0.5  # Reduce influence of low-variance metrics
				if should_ignore_low_variance:
					metric_config["ignore_metric"] = True
					metric_config["removal_reason"] = f"Low variance (std: {std_dev:.6f}, mean: {avg_value:.6f})"
					if verbose:
						console.print(
							f"[yellow]Excluding '{metric}' from analysis due to low variance "
							f"(std: {std_dev:.6f}, mean: {avg_value:.6f}).[/yellow]"
						)

	# Log-transform extreme values in `wavelet_energy_entropy` if necessary
	if 'wavelet_energy_entropy' in results_df.columns:
		if results_df['wavelet_energy_entropy'].min() < 0:
			if verbose:
				console.print(f"[yellow]Log-transforming `wavelet_energy_entropy` due to negative values.[/yellow]")
			results_df['wavelet_energy_entropy'] = np.log1p(np.abs(results_df['wavelet_energy_entropy']))
			for metric_config in ranking_config["metrics"]:
				if metric_config["metric"] == 'wavelet_energy_entropy':
					metric_config["was_log_transformed"] = True

		else:
			if verbose:
				console.print(f"[green]No log-transform needed for `wavelet_energy_entropy`. All values are non-negative.[/green]")

	# Handle complex-valued metrics
	complex_columns = results_df.select_dtypes(include=[np.complex_]).columns
	if len(complex_columns) > 0:
		if verbose:
			console.print(f"[yellow]Handling complex metrics: {list(complex_columns)}[/yellow]")
		for column in complex_columns:
			results_df[column] = np.abs(results_df[column])
	else:
		if verbose:
			console.print(f"[green]No complex-valued metrics found. Skipping this step.[/green]")
	
	return results_df, ranking_config

## WAVELET NORMALIZATION & Z-SCORING FUNCTIONS
def normalize_metrics(
	df: pd.DataFrame, 
	metric_list: list,  
	ranking_config: dict, 
	prefix: str = "",
	feature_range=(0, 1)
) -> tuple:
	"""
	Scales the provided metrics in the DataFrame using RobustScaler to handle outliers,
	then MinMaxScaler for final normalization into a specified feature range (default (0,1)).
	Also applies "lower is better" logic by inverting the scale for such metrics.

	Parameters:
	----------
	df : pd.DataFrame
		DataFrame with raw metrics.
	metric_list : list
		List of metrics to normalize.
	ranking_config : dict, optional
		Dictionary to log normalization details (flags about normalization).
	prefix : str, optional
		Prefix for column names (e.g., "combined_" for merged datasets).
	feature_range : tuple of (float, float), optional
		The (min, max) range for the final MinMax scaling.

	Returns:
	--------
	tuple: (pd.DataFrame, dict)
		1) DataFrame with normalized metrics in columns like [prefix]{metric}_normalized
		2) Updated ranking configuration with normalization details.
	"""
	scaler_robust = RobustScaler()
	scaler_minmax = MinMaxScaler(feature_range=feature_range)

	for metric in metric_list:
		norm_col = f"{prefix}{metric}_normalized"

		try:
			# 1) Robust Scaling to reduce outlier effects
			robust_scaled_values = scaler_robust.fit_transform(df[[metric]]).flatten()

			# 2) MinMax Scaling for final normalization
			minmax_scaled_values = scaler_minmax.fit_transform(
				robust_scaled_values.reshape(-1, 1)
			).flatten()

			# 3) Invert if "lower is better"
			if metric in LOWER_IS_BETTER_METRICS:
				# If the entire column is zero (rare), just set everything to 0
				if df[metric].max() == 0 or pd.isna(df[metric].max()):
					df[norm_col] = 0
				else:
					df[norm_col] = 1 - minmax_scaled_values
			else:
				df[norm_col] = minmax_scaled_values

			# -- Update ranking_config metadata --
			if ranking_config is not None:
				for metric_config in ranking_config["metrics"]:
					if metric_config["metric"] == metric:
						metric_config["was_normalized"] = True
						metric_config["was_inverted"] = metric in LOWER_IS_BETTER_METRICS
						break

		except ValueError as e:
			console.print(f"[red]Skipping normalization for {metric}: {e}[/red]")
			# Assign NaN for debugging
			df[norm_col] = np.nan

			# Log error in ranking_config
			if ranking_config is not None:
				for metric_config in ranking_config["metrics"]:
					if metric_config["metric"] == metric:
						metric_config["ignore_metric"] = True
						metric_config["removal_reason"] = f"Normalization Error: {e}"
						break
	return df, ranking_config

def calculate_normalized_weighted_scores_by_metric_type(df: pd.DataFrame, metric_weights: dict, metric_type: str, prefix, should_calculate_summed_scores: bool = False) -> tuple:
	"""
	This function weights scores for each metric based on the provided weights. It computes both weighted and summed scores for each metric type. It first computes the weighted scores for each metric by multiplying the metric value by the corresponding weight. It then sums the weighted scores across all metrics to obtain the total weighted score for each row. It also computes the simple summation score by summing the raw metric values across all metrics. The function then ranks the results based on the weighted scores and returns the ranked DataFrame.

	Parameters:
	-----------
	df : pd.DataFrame
		DataFrame containing the results with metrics.
	metric_weights : dict
		Configuration dictionary containing metric weights.
	metric_type : str
		Type of metric being weighted (e.g., "signal" or "reconstruction").
	prefix : str, optional
		Prefix for column names (e.g., "combined_" for merged datasets).
	should_calculate_summed_scores : bool, optional
		Flag to calculate summed scores in addition to weighted scores.
	
	Returns:
	--------
	tuple: pd.DataFrame, list
		df_copy : pd.DataFrame
			DataFrame containing the comparison results and rankings.
		columns : list
			List of columns used for comparison.
	"""
	if not metric_weights:  # Ensure metrics exist
		console.print(f"[yellow]No metrics provided for {metric_type}. Skipping calculations.[/yellow]")
		return df.copy(), []
	
	# We'll create a copy of df so we don't modify the original in-place.
	df_copy = df.copy()

	# Name of the columns we'll produce
	weighted_col = f"{prefix}{metric_type}_weighted_score"
	weighted_norm_col = f"{prefix}{metric_type}_normalized_weighted_score"
	weighted_rank_col = f"{prefix}{metric_type}_normalized_weighted_rank"

	# Collect valid columns
	columns = [
		f"{prefix}{metric}" 
		for metric in metric_weights.keys() 
		if f"{prefix}{metric}" in df_copy.columns
	]
	# Calculate weighted scores row-by-row
	tqdm.pandas(desc=f"Calculating {metric_type} weighted scores")
	df_copy[weighted_col] = df_copy.progress_apply(
		lambda row: sum(
			metric_weights[metric] * row[f"{prefix}{metric}"]
			for metric in metric_weights
			if f"{prefix}{metric}" in df_copy.columns 
			   and pd.notna(row[f"{prefix}{metric}"])
		),
		axis=1
	)

	# Normalize weighted scores
	max_score = df_copy[weighted_col].max()
	if pd.notna(max_score) and max_score > 0:
		df_copy[weighted_norm_col] = df_copy[weighted_col] / max_score
	else:
		console.print(f"[yellow]Max score is zero or NaN for {metric_type}. Setting normalized to NaN.[/yellow]")
		df_copy[weighted_norm_col] = np.nan

	# Sort by normalized weighted score for ranking (descending)
	df_copy = df_copy.sort_values(by=weighted_norm_col, ascending=False).reset_index(drop=True)
	df_copy[weighted_rank_col] = df_copy.index + 1

	if should_calculate_summed_scores:
		summed_col = f"{prefix}{metric_type}_summed_score"
		summed_norm_col = f"{prefix}{metric_type}_normalized_summed_score"
		summed_rank_col = f"{prefix}{metric_type}_normalized_summed_rank"

		# Compute a simple sum across all metrics 
		df_copy[summed_col] = df_copy[columns].sum(axis=1)

		# Normalize the summed score 
		max_sum = df_copy[summed_col].max()
		if pd.notna(max_sum) and max_sum > 0:
			df_copy[summed_norm_col] = df_copy[summed_col] / max_sum
		else:
			console.print(f"[yellow]Max summed score is zero or NaN for {metric_type}. Setting to NaN.[/yellow]")
			df_copy[summed_norm_col] = np.nan
	
		df_copy = df_copy.sort_values(by=summed_norm_col, ascending=False).reset_index(drop=True)
		df_copy[summed_rank_col] = df_copy.index + 1

	return df_copy, columns

def normalize_weights_dynamically(
	metric_columns: list,    # e.g. ["relative_peaks_normalized", "positive_frequencies_normalized"]
	base_weights: dict,      # e.g. {"relative_peaks": 0.2, "positive_frequencies": 0.35}
	df: pd.DataFrame,
	ranking_config: dict,
	presence_threshold: float = 0.1,
	variance_threshold: float = 1e-6,
	min_weight: float = 0.01,
	penalize_low_variance: bool = False,
	penalty_factor: float = 0.5,
) -> tuple:
	"""
	Dynamically normalizes weights for metrics based on variance and presence thresholds.
	This function:
	1. Computes presence and variance for each metric.
	2. Excludes metrics below the defined thresholds.
	3. Optionally penalizes low-variance metrics.
	4. Normalizes final weights to sum to 1.
	5. Returns updated ranking_config and **a list of valid metrics**.

	Parameters
	----------
	metric_columns : list of str
		Full column names in `df` (e.g., 'relative_peaks_normalized').
	base_weights : dict
		Original weights keyed by the base metric name (e.g. 'relative_peaks').
	df : pd.DataFrame
		DataFrame containing the columns in metric_columns.
	ranking_config : dict
		Config with flags like 'ignore_metric', 'low_variance_detected', etc.
	presence_threshold : float
		If a metric's presence < this threshold, we exclude it.
	variance_threshold : float
		If a metric's variance < this threshold, we exclude it.
	min_weight : float
		Minimum weight to assign if a metric passes checks.
	penalize_low_variance : bool
		If True, apply penalty_factor to metrics flagged low_variance_detected.
	penalty_factor : float
		The factor by which to reduce the weight of flagged low-variance metrics (e.g., 0.5).

	Returns
	-------
	(final_weights, updated_ranking_config, valid_metrics)
		final_weights : dict
			A dict mapping the *full column name* (e.g., 'relative_peaks_normalized')
			to its final normalized weight.
		updated_ranking_config : dict
			The updated ranking config, logging final weights.
		valid_metrics : list
			List of metric columns that passed presence/variance checks.
	"""
	# Compute presence & variance
	metric_presence = df[metric_columns].notna().mean()
	metric_variance = df[metric_columns].var()

	final_weights = {}
	valid_metrics = []

	# Iterate over each valid metric
	for full_col_name in metric_columns:
		base_name = full_col_name.replace("_normalized", "")

		# Check presence & variance
		pres_val = metric_presence[full_col_name]
		var_val = metric_variance[full_col_name]

		if pres_val < presence_threshold or var_val < variance_threshold:
			# Flag metric for exclusion
			for mc in ranking_config["metrics"]:
				if mc["metric"] == base_name:
					mc["ignore_metric"] = True
					mc["removal_reason"] = f"Presence: {pres_val:.2f}, Variance: {var_val:.6f}"
					if pres_val < presence_threshold:
						mc["removal_reason"] += f" (below {presence_threshold})"
					if var_val < variance_threshold:
						mc["removal_reason"] += f" (below {variance_threshold})"
			continue  # Skip this metric

		# Retrieve original weight and apply adjustment
		orig_weight = base_weights.get(base_name, min_weight)
		adjusted = max(min_weight, orig_weight * pres_val * var_val)

		# Optionally penalize low variance
		if penalize_low_variance:
			for mc in ranking_config["metrics"]:
				if mc["metric"] == base_name and mc.get("low_variance_detected", False):
					adjusted *= penalty_factor

		final_weights[full_col_name] = adjusted
		valid_metrics.append(full_col_name)  # Track this metric as valid

	# Check if any valid weights remain
	total_w = sum(final_weights.values())

	# If no metrics survived, return empty dictionary
	if total_w == 0 or len(valid_metrics) == 0:
		console.print("[red]All metrics removed by presence/variance thresholds or ignore flags (two-pass).[/red]")
		return {}, ranking_config, []

	# Normalize sum -> 1
	for col_name in final_weights:
		final_weights[col_name] /= total_w

	# Update ranking_config with final weights
	for col_name, w in final_weights.items():
		base_name = col_name.replace("_normalized", "")
		for mc in ranking_config["metrics"]:
			if mc["metric"] == base_name:
				mc["final_weight"] = w
				mc["normalized_weight"] = w
				break

	return final_weights, ranking_config, valid_metrics

def calculate_dynamically_normalized_weighted_score_by_metric_type(
	normalized_df: pd.DataFrame,
	updated_metrics: list,
	ranking_config: dict,
	weights: dict,
	prefix: str,
	metric_type: str,
	epsilon_threshold: float = 1e-6,
	penalty_weight: float = 0.05
) -> tuple:
	"""
	Computes a single dynamic score for the given metric_type using a two-pass approach:
	  1) Calls 'normalize_weights_dynamically' to get final per-metric weights,
	  2) Optionally penalizes rows for missing metrics,
	  3) Sorts by the final dynamic score and returns a ranked DataFrame.

	Parameters
	----------
	normalized_df : pd.DataFrame
		DataFrame containing *normalized* metric columns (e.g., 'relative_peaks_normalized').
	updated_metrics : list of str
		List of metric columns used in the final weighting step (e.g. 'relative_peaks_normalized').
	ranking_config : dict
		Dictionary containing ranking configuration (with ignore_metric, etc.).
	weights : dict
		Original dictionary of static metric weights (keyed by base metric name).
	prefix : str
		Prefix for columns (e.g. "combined_").
	metric_type : str
		The type of metric being processed (e.g., "reconstruction" or "signal").
	epsilon_threshold : float, optional
		Small value to avoid division-by-zero in dynamic score.
	penalty_weight : float, optional
		Weight factor for penalizing missing metrics.

	Returns
	-------
	(ranked_results : pd.DataFrame, updated_ranking_config : dict)
	"""
	# 1. Dynamically compute final weights (two-pass approach)
	normalized_weights, updated_ranking_config, finalized_metrics = normalize_weights_dynamically(
		metric_columns=updated_metrics,
		base_weights=weights,
		df=normalized_df,
		ranking_config=ranking_config
		# Add threshold overrides if you need custom presence_threshold, etc.
	)

	if normalized_weights == {}:
		console.print("[yellow]All metrics removed by presence/variance thresholds or ignore flags (two-pass). Skipping dynamic weighting.[/yellow]")
		return normalized_df, updated_ranking_config

	# 2. Optional: Compute penalty for missing metrics
	normalized_df[f"{prefix}{metric_type}_num_missing_metrics"] = normalized_df[finalized_metrics].isna().sum(axis=1)

	metric_lookup = {m["metric"]: m.get("ignore_metric", False) for m in updated_ranking_config["metrics"]}
	missing_metric_col = f"{prefix}{metric_type}_missing_metrics_count"
	normalized_df[missing_metric_col] = normalized_df.apply(
		lambda row: sum(
			normalized_weights.get(col, 0) * (1 if pd.isna(row[col]) else 0.5)
			for col in finalized_metrics
			if col in normalized_df.columns
			   and (pd.isna(row[col]) or metric_lookup.get(col.replace("_normalized", ""), False))
		),
		axis=1
	)

	# 3. Compute main dynamic score
	dynamic_score_col = f"{prefix}{metric_type}_dynamically_weighted_score"
	ranking_config["penalty_weight"] = penalty_weight

	tqdm.pandas(desc=f"Calculating {metric_type} dynamic scores")
	normalized_df[dynamic_score_col] = normalized_df.progress_apply(
		lambda row: (
			sum(
				normalized_weights[col] * row[col]
				for col in finalized_metrics
				if col in normalized_weights and pd.notna(row[col])
			)
			/ max(
				sum(normalized_weights[col] for col in finalized_metrics if pd.notna(row[col])), 
				epsilon_threshold
			)
			- penalty_weight * row[missing_metric_col]
		),
		axis=1
	)

	# 4. Normalize that score to [0,1]
	dynamic_norm_col = f"{prefix}{metric_type}_normalized_dynamically_weighted_score"
	max_score = normalized_df[dynamic_score_col].max()
	if max_score > 0:
		normalized_df[dynamic_norm_col] = normalized_df[dynamic_score_col] / max_score
	else:
		console.print(f"[yellow]Max dynamic score is zero or NaN! Assigning equal scores for {metric_type}.[/yellow]")
		normalized_df[dynamic_norm_col] = 1.0

	# 6. Rank by the final dynamic score
	ranked_results = normalized_df.sort_values(
		by=dynamic_norm_col, ascending=False
	).reset_index(drop=True)
	ranked_results[f"{prefix}{metric_type}_normalized_dynamically_weighted_rank"] = ranked_results.index + 1

	return ranked_results, updated_ranking_config

def update_ranking_config(ranked_results: pd.DataFrame, ranking_config: dict, prefix: str = "") -> dict:
	"""
	Updates the ranking configuration with computed statistics for each metric.
	This includes min, max, mean, std deviation, variance, correlation with final scores,
	and dynamically normalized values.

	Parameters:
	----------
	ranked_results : pd.DataFrame
		DataFrame containing the final ranking results with weighted and normalized scores.
	ranking_config : dict
		Dictionary storing ranking metadata.
	prefix : str, optional
		Prefix for column names (e.g., "combined_" for merged datasets).

	Returns:
	--------
	ranking_config : dict
		Updated ranking configuration with additional statistics.
	"""

	# Iterate through all the metrics in the ranking config
	for metric_config in ranking_config["metrics"]:
		metric = metric_config["metric"]

		if metric in ranked_results.columns:
			# Store raw metric statistics
			metric_config.update({
				f"{prefix}mean": ranked_results[metric].mean(),
				f"{prefix}std": ranked_results[metric].std(),
				f"{prefix}max": ranked_results[metric].max(),
				f"{prefix}min": ranked_results[metric].min(),
			})

		if f"{prefix}{metric}_normalized" in ranked_results.columns:
			# Store normalized metric statistics
			metric_config.update({
				f"mean_normalized": ranked_results[f"{prefix}{metric}_normalized"].mean(),
				f"std_normalized": ranked_results[f"{prefix}{metric}_normalized"].std(),
				f"max_normalized": ranked_results[f"{prefix}{metric}_normalized"].max(),
				f"min_normalized": ranked_results[f"{prefix}{metric}_normalized"].min(),
			})

	# Store overall statistics for key scores
	for score_type in ["weighted", "dynamic"]:
		for metric_type in ["reconstruction", "signal"]:
			score_col = f"{prefix}{metric_type}_normalized_{score_type}_score"
			if score_col in ranked_results.columns:
				ranking_config[f"{metric_type}_normalized_{score_type}_max"] = ranked_results[score_col].max()
				ranking_config[f"{metric_type}_normalized_{score_type}_min"] = ranked_results[score_col].min()
				ranking_config[f"{metric_type}_normalized_{score_type}_mean"] = ranked_results[score_col].mean()
				ranking_config[f"{metric_type}_normalized_{score_type}_std"] = ranked_results[score_col].std()

	return ranking_config

def calculate_rank_stability(
	df: pd.DataFrame, 
	rank_columns: list, 
	prefix: str, 
	weight_factor: float = 0.5
) -> pd.DataFrame:
	"""
	Calculate a simplified stability metric for wavelet rankings based on multiple ranking columns.
	This function focuses on:
	  - rank_variability (standard deviation of ranks across columns)
	  - normalized_rank_stability (higher = less variation)
	  - average_rank (mean of rank columns, where lower rank is better)
	  - harmonic_average_rank (optional, for reference)
	  - stability_score_weighted (final combined metric, where higher = more stable)

	Parameters:
	-----------
	df : pd.DataFrame
		DataFrame containing rank columns to evaluate.
	rank_columns : list of str
		Columns representing ranks to compare for stability.
	prefix : str
		Prefix for column names (e.g., "combined_" for merged datasets).
	weight_factor : float, optional
		Weight factor (in [0, 1]) controlling how we combine normalized_rank_stability 
		vs. inverted average_rank in the final stability score. 
		(Default is 0.5, giving them equal weight.)

	Returns:
	--------
	pd.DataFrame
		DataFrame with additional stability metrics and a final stability rank.
	"""
	# 1. Calculate rank variability (standard deviation across the rank columns).
	df[f"{prefix}rank_variability"] = df[rank_columns].std(axis=1)

	# 2. Compute normalized_rank_stability = 1 - (rank_variability / max_rank).
	#    Here, if rank_variability is small, stability is close to 1; if large, close to 0.
	max_rank = df[rank_columns].max().max()
	df[f"{prefix}normalized_rank_stability"] = 1 - (
		df[f"{prefix}rank_variability"] / max_rank
	)

	# 3. Compute average_rank (mean across the rank columns).
	#    Typically, a lower average rank indicates "better" overall rank.
	df[f"{prefix}average_rank"] = df[rank_columns].mean(axis=1)

	# 4. (Optional) Harmonic mean of ranks, if you like to see it. 
	#    If you don't need it, comment or remove these lines:
	df[f"{prefix}harmonic_average_rank"] = len(rank_columns) / np.sum(1 / df[rank_columns], axis=1)

	# 5. Define a final stability score, stability_score_weighted,
	#    where "higher" = "more stable."
	#    - We combine normalized_rank_stability (bigger is better)
	#    - and (1 - average_rank / max_rank) so that a lower average rank becomes bigger.
	#    weight_factor controls the emphasis between the two.
	df[f"{prefix}stability_score_weighted"] = (
		weight_factor * df[f"{prefix}normalized_rank_stability"]
		+ (1 - weight_factor) * (1 - (df[f"{prefix}average_rank"] / max_rank))
	)

	# 6. Sort by stability_score_weighted descending (highest = best stability)
	df = df.sort_values(
		by=f"{prefix}stability_score_weighted", ascending=False
	).reset_index(drop=True)

	# 7. Assign final stability_rank (lowest index => rank #1).
	df[f"{prefix}stability_rank"] = df.index + 1

	return df

def calculate_wavelet_family_stability(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
	"""
	Computes wavelet family-level rank stability based on the final wavelet-level ranks,
	then merges it back to each individual wavelet record.

	Parameters
	----------
	df : pd.DataFrame
		DataFrame with wavelet rankings and stability metrics. 
		It must have columns like:
		  - "wavelet": the wavelet name (e.g. "db4", "coif3", etc.)
		  - f"{prefix}stability_rank": the final stability rank across wavelets
		If other columns (e.g. average rank) are present and relevant, we can aggregate them too.
	prefix : str
		Prefix for metric columns (e.g., "combined_" for merged datasets).

	Returns
	-------
	pd.DataFrame
		DataFrame with both individual wavelet rankings and family-level stability scores.
	"""

	# 1. Extract wavelet families by parsing the "wavelet" column.
	#    This typically means "db4" => wavelet_family="db", "coif3" => wavelet_family="coif", etc.
	if 'wavelet_family' not in df.columns:
		df["wavelet_family"] = (
			df["wavelet"].str.extract(r"([a-zA-Z]+)").fillna("Unknown")
		)

	# 2. Compute family-level stats based on final wavelet ranks.
	#    We'll assume the relevant final rank is f"{prefix}stability_rank",
	#    but adapt if you want e.g. f"{prefix}final_dynamic_rank" instead.
	family_rank_stats = df.groupby("wavelet_family").agg(
		family_mean_stability_rank=(f"{prefix}stability_rank", "mean"),
		family_median_stability_rank=(f"{prefix}stability_rank", "median"),
		family_rank_std_dev=(f"{prefix}stability_rank", "std"),
		family_wavelet_count=("wavelet", "count")
	).reset_index()

	# Normalize the columns we care about
	max_mean = family_rank_stats["family_mean_stability_rank"].max()
	max_std  = family_rank_stats["family_rank_std_dev"].max()

	# Avoid zero-division
	if max_mean == 0:
		max_mean = 1e-12
	if pd.isna(max_std) or max_std == 0:
		# If all wavelets in a family have the same rank, std=0 => skip or set to small
		max_std = 1e-12

	family_rank_stats[f"{prefix}normalized_family_mean_rank"] = (
		family_rank_stats["family_mean_stability_rank"] / max_mean
	)
	family_rank_stats[f"{prefix}normalized_family_std_dev"] = (
		family_rank_stats["family_rank_std_dev"] / max_std
	)

	# Weighted combination, adjusting as desired
	# Example: place 70% emphasis on a low mean rank, 30% emphasis on a low std dev
	family_rank_stats[f"{prefix}final_family_stability_score"] = (
		0.7 * (1 - family_rank_stats[f"{prefix}normalized_family_mean_rank"]) 
		+ 0.3 * (1 - family_rank_stats[f"{prefix}normalized_family_std_dev"])
	)

	# 4. Rank families by the final family stability score (descending => best at top)
	family_rank_stats = family_rank_stats.sort_values(
		by=f"{prefix}final_family_stability_score", ascending=False
	).reset_index(drop=True)
	family_rank_stats[f"{prefix}final_family_stability_rank"] = family_rank_stats.index + 1

	# 5. Merge back with the original DataFrame
	drop_cols = list(set(df.columns) & set(family_rank_stats.columns))
	drop_cols = [col for col in drop_cols if col != "wavelet_family"]
	console.print(f"[cyan]Dropping columns: {drop_cols}[/cyan]")
	df = df.drop(columns=drop_cols, errors="ignore")
	merge_cols = ["wavelet_family"]
	df = df.merge(family_rank_stats, on=merge_cols, how="left")

	# 6. Optionally define a "family_informed_rank" if you want to combine wavelet-level rank
	#    with family-level rank. For example:
	df[f"{prefix}family_informed_rank"] = (
		0.75 * df[f"{prefix}stability_rank"] 
		+ 0.25 * df[f"{prefix}final_family_stability_rank"]
	)

	return df

def select_top_ranked_results(
	ranked_results: pd.DataFrame,
	prefix: str,
	ranking_config: dict,
	percentage_of_results: float = 0.1
) -> tuple:
	"""
	Selects the top-ranked wavelet configurations dynamically based on a chosen rank column.
	Ensures that the best-performing configurations are chosen within the top N% of results,
	while also selecting the best configuration for each wavelet group (to ensure coverage).

	Parameters
	----------
	ranked_results : pd.DataFrame
		DataFrame containing ranked results with computed scores.
	prefix : str
		Prefix for column names (e.g., "combined_" for merged datasets).
	ranking_config : dict
		Configuration dictionary to store selection details.
	percentage_of_results : float, optional
		Fraction of results to retain (default is 10%).

	Returns
	-------
	(final_ranked_results : pd.DataFrame, ranking_config : dict)
		final_ranked_results : DataFrame with the final top-ranked wavelet configurations.
		ranking_config : Updated configuration with details on which rank column was used.
	"""

	# --- Step 1: Check if family_informed_rank exists, then check correlation with stability_rank
	stability_col = f"{prefix}stability_rank"
	family_col = f"{prefix}family_informed_rank"

	if stability_col not in ranked_results.columns:
		console.print(f"[red]'{stability_col}' not found in ranked_results. Using only existing columns.[/red]")
		rank_col = stability_col  # fallback
	elif family_col in ranked_results.columns:
		# Compute correlation
		rank_corr = ranked_results[[stability_col, family_col]].corr().iloc[0, 1]
		console.print(f"[cyan]Correlation between {stability_col} and {family_col}: {rank_corr:.3f}[/cyan]")

		# Decide on rank column
		if rank_corr >= 0.85:
			rank_col = family_col
			console.print("[green]Using family-informed rank for selection (strong correlation).[/green]")
		else:
			rank_col = stability_col
			console.print("[yellow]Using stability rank (weak correlation with family rank).[/yellow]")

		ranking_config[f"{prefix}selected_family_rank_stability_column"] = rank_col
		ranking_config[f"{prefix}selected_family_rank_stability_column_correlation"] = rank_corr
	else:
		# If there's no family_informed_rank column, default to stability_rank
		rank_col = stability_col
		console.print("[green]No family_informed_rank column found. Using stability_rank for selection.[/green]")

	# --- Step 2: Select Top N% Based on Chosen Rank
	num_top_results = max(1, int(len(ranked_results) * percentage_of_results))
	# rank_col: lower = better
	top_ranked_results = ranked_results.nsmallest(num_top_results, rank_col)
	console.print(f"[cyan]Selecting top {num_top_results} results by '{rank_col}'.[/cyan]")

	# --- Step 3: Also select the best configuration per wavelet group
	# Adjust grouping columns based on your data schema
	# If 'wavelet_type' doesn't exist, just group by "wavelet"
	grouping_cols = []
	if 'wavelet_type' in ranked_results.columns:
		grouping_cols.append('wavelet_type')
	if 'wavelet' in ranked_results.columns:
		grouping_cols.append('wavelet')

	if not grouping_cols:
		console.print("[red]No wavelet grouping columns found. Will skip best-per-group selection.[/red]")
		subset_ranked_results = pd.DataFrame()
	else:
		grouped = ranked_results.groupby(grouping_cols, as_index=False)
		# For each group, pick the row with the smallest (best) rank_col
		subset_ranked_results = grouped.apply(
			lambda group: group.loc[group[rank_col].idxmin()]
		).reset_index(drop=True)

	# --- Step 4: Combine & Deduplicate
	# We define some columns to check for duplication
	drop_cols = []
	for col in ['signal_type', 'wavelet_type', 'wavelet', 'wavelet_mode', 'wavelet_level']:
		if col in subset_ranked_results.columns and col in top_ranked_results.columns:
			drop_cols.append(col)

	final_ranked_results = pd.concat([top_ranked_results, subset_ranked_results], ignore_index=True)
	final_ranked_results = final_ranked_results.drop_duplicates(subset=drop_cols, keep='first')

	# --- Step 5: Sort by the chosen rank column to assign final rank
	final_ranked_results = final_ranked_results.sort_values(by=rank_col).reset_index(drop=True)
	final_ranked_results[f"{prefix}top_wavelet_rank"] = final_ranked_results.index + 1

	return final_ranked_results, ranking_config

def determine_best_wavelet_representation(
	results_df: pd.DataFrame, signal_type: str, original_signal_metrics_df: pd.DataFrame, prefix: str = "", epsilon_threshold: float = 1e-6, penalty_weight: float = 0.05, percentage_of_results: float = 0.1, ignore_low_variance: bool = True
) -> tuple:
	# Generate ranking configuration
	ranking_config = generate_ranking_config(signal_type, prefix)

	if len(prefix) == 0:
		console.print("Calculating reconstructed and signal metrics...", style="bright_white")
		initial_preprocessed_results_df = preprocess_reconstructed_metrics(original_signal_metrics_df, results_df)
		preprocessed_results_df, ranking_config = preprocess_signal_metrics(initial_preprocessed_results_df, ranking_config, ignore_low_variance, epsilon_threshold)

	else:
		console.print("Calculating signal metrics...", style="bright_white")
		preprocessed_results_df, ranking_config = preprocess_signal_metrics(results_df, ranking_config, ignore_low_variance, epsilon_threshold)
	
	metrics = [
		metric_config["metric"]
		for metric_config in ranking_config["metrics"]
		if not metric_config.get("ignore_metric", False)  # Exclude ignored metrics
		and metric_config["metric"] in preprocessed_results_df.columns  # Ensure existence in `results_df`
	]
	
	# Normalize and Z-score metrics
	normalized_results_df, ranking_config = normalize_metrics(preprocessed_results_df, metrics, ranking_config, prefix)


	# Calculate weighted normalized scores
	partial_weighted_scored_df, reconstruction_columns = calculate_normalized_weighted_scores_by_metric_type(normalized_results_df, RECONSTRUCTION_METRIC_WEIGHTS, "reconstruction", prefix)
	full_weighted_scored_df, signal_columns = calculate_normalized_weighted_scores_by_metric_type(partial_weighted_scored_df, SIGNAL_METRIC_WEIGHTS, "signal", prefix)

	updated_reconstruction_metrics = [
		metric_config["metric"] + "_normalized"
		for metric_config in ranking_config["metrics"]
		if not metric_config.get("ignore_metric", False)  # Exclude ignored metrics
		and metric_config["metric"] + "_normalized" in reconstruction_columns  # Ensure existence in `results_df`
	]

	updated_signal_metrics = [
		metric_config["metric"] + "_normalized"
		for metric_config in ranking_config["metrics"]
		if not metric_config.get("ignore_metric", False)  # Exclude ignored metrics
		and metric_config["metric"] + "_normalized" in signal_columns  # Ensure existence in `results_df`
	]
	

	# Normalize weights dynamically
	dynamically_ranked_results, updated_ranking_config = calculate_dynamically_normalized_weighted_score_by_metric_type(full_weighted_scored_df, updated_reconstruction_metrics, ranking_config, RECONSTRUCTION_METRIC_WEIGHTS, prefix, "reconstruction", epsilon_threshold, penalty_weight)
	final_ranked_results, final_ranking_config = calculate_dynamically_normalized_weighted_score_by_metric_type(dynamically_ranked_results, updated_signal_metrics, updated_ranking_config, SIGNAL_METRIC_WEIGHTS, prefix, "signal", epsilon_threshold, penalty_weight)

	# Update ranking configuration with additional statistics
	final_ranking_config = update_ranking_config(final_ranked_results, final_ranking_config, prefix)


	# **Compute Correlation Between Different Ranks**
	rank_cols = [
		f"{prefix}reconstruction_normalized_weighted_rank", f"{prefix}reconstruction_normalized_dynamically_weighted_rank", 
		f"{prefix}signal_normalized_weighted_rank", f"{prefix}signal_normalized_dynamically_weighted_rank",
	]
	final_rank_cols = [col for col in rank_cols if col in final_ranked_results.columns]

	final_ranking_config["ranking_columns"] = final_rank_cols
	
	# **Calculate Rank Stability Using Selected Columns**
	stable_ranked_results = calculate_rank_stability(final_ranked_results, final_rank_cols, prefix)

	total_ranked_results = calculate_wavelet_family_stability(stable_ranked_results, prefix)
	
	top_ranked_results, final_ranking_config = select_top_ranked_results(total_ranked_results, prefix, final_ranking_config, percentage_of_results)

	return top_ranked_results, total_ranked_results, final_ranking_config