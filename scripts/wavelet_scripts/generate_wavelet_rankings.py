# Standard library imports
import warnings
from typing import Any, Tuple, Union

# Third-party imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from rich.console import Console
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.stats import wasserstein_distance
from difflib import SequenceMatcher
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
	"relative_right_bases_matcher_alignment_score_normalized": 0.1,
	"relative_left_bases_global_alignment_score_normalized": 0.1,
	"relative_left_bases_matcher_alignment_score_normalized": 0.1,
	"upper_envelope_diff_normalized": 0.2,
	"lower_envelope_diff_normalized": 0.2,
	"relative_peaks_matcher_alignment_score_normalized": 0.2,
	"relative_peaks_global_alignment_score_normalized": 0.2,
}

LOWER_IS_BETTER_METRICS = [
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
	"dynamic_cutoff_diff_normalized",
	"frequency_max_diff_normalized",
	"spectral_bandwidth_diff_normalized",
	"num_fft_peaks_diff_normalized",
	"relative_num_peaks_diff_normalized",
	"dominant_frequency_diff_normalized",
	"spectral_magnitude_diff_normalized",
	"spectral_centroid_diff_normalized",
	"relative_right_bases_global_alignment_score_normalized",
	"relative_right_bases_matcher_alignment_score_normalized",
	"relative_left_bases_global_alignment_score_normalized",
	"relative_left_bases_matcher_alignment_score_normalized",
	"relative_peaks_matcher_alignment_score_normalized",
	"relative_peaks_global_alignment_score_normalized",
	"upper_envelope_diff_normalized",
	"lower_envelope_diff_normalized",
	"kl_divergence_norm",
	"emd_value_norm",
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
def run_global_sequence_alignment(original_sequence: list, reconstructed_sequence: list, placeholder: int = -1) -> int:
	"""
	Applies global sequence alignment on the implied zero values within a reconstructed_sequence using minineedle, with placeholders.
	
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
		The alignment score. If the alignment fails, returns 0. A higher score indicates a better alignment.
	"""
	# Extract the observed sequence from the reconstructed_sequence
	observed_sequence = [float(p) if pd.notna(p) else placeholder for p in reconstructed_sequence]
	
	# Check for valid entries in the observed sequence
	if all(val == placeholder for val in observed_sequence):
		return 0

	# Create Needleman-Wunsch global alignment instance
	alignment = needle.NeedlemanWunsch(observed_sequence, original_sequence)
	alignment.change_matrix(core.ScoreMatrix(match=4, miss=-0.5, gap=-1))

	try:
		# Run the alignment
		alignment.align()
		alignment_score = alignment.get_score()
		return alignment_score

	except ZeroDivisionError:
		return 0

def align_arrays(original_array: list, reconstructed_array: list) -> float:
	"""
	Align the original and reconstructed arrays using sequence alignment. Uses the SequenceMatcher from difflib to compare the two arrays and return a similarity ratio. A higher ratio indicates a better alignment.

	Parameters
	----------
	original_array : list
		The original array to be aligned.
	reconstructed_array : list
		The reconstructed array to be aligned.

	Returns
	-------
	float
		The alignment score. A higher score indicates a better alignment.
	"""
	# Convert to strings for alignment
	original_array_str = " ".join(map(str, original_array))
	reconstructed_array_str = " ".join(map(str, reconstructed_array))
	matcher = SequenceMatcher(None, original_array_str, reconstructed_array_str)
	return matcher.ratio()

def compare_distributions(original_array: list, reconstructed_array: list, method: str = "dtw") -> float:
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
	try:
		if method == "dtw":
			distance, _ = fastdtw(original_array, reconstructed_array)
			return distance
		elif method == "euclidean":
			if len(original_array) != len(reconstructed_array):
				raise ValueError("Euclidean distance requires arrays of equal length.")
			return np.sqrt(np.sum((original_array - reconstructed_array) ** 2))
		elif method == "wasserstein":
			return wasserstein_distance(original_array, reconstructed_array)
		else:
			raise ValueError(f"Unknown method: {method}")
	except Exception as e:
		console.print(f"[red]Error comparing distributions using method '{method}': {e}[/red]")
		return np.nan

def compare_prominences_distribution(original_array: list, reconstructed_array: list) -> tuple:
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
	avg_diff = np.abs(np.mean(original_array) - np.mean(reconstructed_array))
	total_diff = np.abs(np.sum(original_array) - np.sum(reconstructed_array))
	wasserstein = compare_distributions(original_array, reconstructed_array, method="wasserstein")
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
		original_array = (
			np.array(eval(original_value)) if isinstance(original_value, str) and original_value.startswith("[") else np.array(original_value)
		)
		reconstructed_array = (
			np.array(eval(reconstructed_value)) if isinstance(reconstructed_value, str) and reconstructed_value.startswith("[") else np.array(reconstructed_value)
		)
		return original_array, reconstructed_array
	except Exception as e:
		console.print(f"[red]Error processing arrays: {e}[/red]")
		return None, None
	
def convert_to_serializable(value):
	"""Converts NumPy types to Python native types for JSON serialization."""
	if isinstance(value, (np.int64, np.int32)):
		return int(value)
	elif isinstance(value, (np.float64, np.float32, np.float16)):
		return float(value)
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

	
	metrics = set(original_df.columns).intersection(set(reconstructed_df.columns)) - {"signal_type"}

	cols_to_add = ['wavelet', 'wavelet_level','wavelet_mode']
	cols_to_add = [col for col in cols_to_add if col in reconstructed_df.columns]

	for idx, reconstructed_row in tqdm(reconstructed_df.iterrows(), total=reconstructed_df.shape[0], desc="Comparing metrics"):
		for metric in metrics:
			if metric in original_df.columns and metric in reconstructed_df.columns:
				# Check if the metric is list-like or stringified array
				if isinstance(original_df[metric].iloc[0], (list, np.ndarray)) or (
					isinstance(original_df[metric].iloc[0], str) and original_df[metric].iloc[0].startswith("[")
				):
					# Convert to numpy arrays
					original_array, reconstructed_array = process_array(original_df[metric].iloc[0], reconstructed_row[metric])
					# Skip if arrays are None
					if original_array is None or reconstructed_array is None:
						continue
					
					if metric in {"relative_peaks", "relative_left_bases", "relative_right_bases"}:
						reconstructed_df.loc[idx, f"{metric}_matcher_alignment_score"] = align_arrays(original_array, reconstructed_array)
						reconstructed_df.loc[idx,f"{metric}_global_alignment_score"] = run_global_sequence_alignment(original_array, reconstructed_array)
					elif metric in {"positive_frequencies", "positive_amplitudes"}:
						reconstructed_df.loc[idx,f"{metric}_dtw"] = compare_distributions(original_array, reconstructed_array, method="dtw")
						if len(original_array) == len(reconstructed_array):
							reconstructed_df.loc[idx,f"{metric}_euclidean"] = compare_distributions(original_array, reconstructed_array, method="euclidean")
						reconstructed_df.loc[idx,f"{metric}_wasserstein"] = compare_distributions(original_array, reconstructed_array, method="wasserstein")
					elif metric == "relative_prominences":
						avg_diff, total_diff, wasserstein = compare_prominences_distribution(original_array, reconstructed_array)
						reconstructed_df.loc[idx,f"{metric}_avg_diff"] = avg_diff
						reconstructed_df.loc[idx,f"{metric}_total_diff"] = total_diff
						reconstructed_df.loc[idx,f"{metric}_wasserstein"] = wasserstein
				else:
					try:
						reconstructed_df.loc[idx,f"{metric}_diff"] = abs(float(original_df[metric].iloc[0]) - float(reconstructed_row[metric]))
					except Exception as e:
						console.print(f"[red]Error computing difference for metric '{metric}': {e}[/red]")
						reconstructed_df.loc[idx,f"{metric}_diff"] = np.nan
	return reconstructed_df

def preprocess_signal_metrics(results_df: pd.DataFrame, ranking_config: dict,  ignore_low_variance: bool, epsilon_threshold: float = 1e-6) -> tuple:
	"""
	Preprocesses the raw signal metrics DataFrame by handling zero or near-zero variance metrics, log-transforming extreme values in `wavelet_energy_entropy`, and handling complex-valued metrics. It dynamically adjusts weights for low-variance metrics and logs the changes in the ranking configuration. It also log-transforms `wavelet_energy_entropy` if negative values are present. The function then returns the cleaned DataFrame and updated ranking configuration.

	Parameters:
	----------
	results_df : pd.DataFrame
		DataFrame containing raw signal metrics.
	ranking_config : dict
		Dictionary containing the ranking configuration.
	ignore_low_variance : bool
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
				console.print(
					f"[yellow]Low variance for '{metric}' (std: {std_dev:.6f}, mean: {avg_value:.6f}). Adjusting weight.[/yellow]"
				)
				metric_config["ignore_metric"] = False
				metric_config["removal_reason"] = f"Low variance (std: {std_dev:.6f}, mean: {avg_value:.6f})"
				# signal_metric_weights[metric] *= 0.5  # Reduce influence of low-variance metrics
				if ignore_low_variance:
					metric_config["ignore_metric"] = True
					metric_config["removal_reason"] = f"Low variance (std: {std_dev:.6f}, mean: {avg_value:.6f})"
					console.print(
						f"[yellow]Excluding '{metric}' from analysis due to low variance "
						f"(std: {std_dev:.6f}, mean: {avg_value:.6f}).[/yellow]"
					)

	# Log-transform extreme values in `wavelet_energy_entropy` if necessary
	if 'wavelet_energy_entropy' in results_df.columns:
		if results_df['wavelet_energy_entropy'].min() < 0:
			console.print(f"[yellow]Log-transforming `wavelet_energy_entropy` due to negative values.[/yellow]")
			results_df['wavelet_energy_entropy'] = np.log1p(np.abs(results_df['wavelet_energy_entropy']))
			for metric_config in ranking_config["metrics"]:
				if metric_config["metric"] == 'wavelet_energy_entropy':
					metric_config["was_log_transformed"] = True

		else:
			console.print(f"[green]No log-transform needed for `wavelet_energy_entropy`. All values are non-negative.[/green]")

	# Handle complex-valued metrics
	complex_columns = results_df.select_dtypes(include=[np.complex_]).columns
	if len(complex_columns) > 0:
		console.print(f"[yellow]Handling complex metrics: {list(complex_columns)}[/yellow]")
		for column in complex_columns:
			results_df[column] = np.abs(results_df[column])
	else:
		console.print(f"[green]No complex-valued metrics found. Skipping this step.[/green]")
	
	return results_df, ranking_config

## WAVELET NORMALIZATION & Z-SCORING FUNCTIONS
def normalize_and_zscore(df: pd.DataFrame, metric_list: list,  ranking_config: dict, prefix: str = "",
) -> tuple:
	"""
	Normalizes and Z-scores the provided metrics in the DataFrame. It uses a RobustScaler to handle outliers and MinMaxScaler for normalization. The function also logs normalization details in the ranking configuration if provided. It skips normalization for metrics with errors and logs the details in the ranking configuration.

	Parameters:
	----------
	df : pd.DataFrame
		DataFrame with raw metrics.
	metric_list : list
		List of metrics to normalize.
	lower_is_better_metrics : list
		List of metrics where lower values are better (to be inverted).
	prefix : str, optional
		Prefix for column names (e.g., "combined_" for merged datasets).
	ranking_config : dict, optional
		Dictionary to log normalization details.

	Returns:
	--------
	tuple: pd.DataFrame, dict
	pd.DataFrame
		DataFrame with normalized and Z-scored metrics.
	dict
		Updated ranking configuration with normalization details.
	"""

	scaler_robust = RobustScaler()
	scaler_minmax = MinMaxScaler(feature_range=(0, 1))

	for metric in metric_list:
		norm_col = f"{prefix}{metric}_normalized"
		zscore_col = f"{prefix}{metric}_zscore"

		try:
			# Robust Scaling to handle outliers
			robust_scaled_values = scaler_robust.fit_transform(df[[metric]]).flatten()
			minmax_scaled_values = scaler_minmax.fit_transform(robust_scaled_values.reshape(-1, 1)).flatten()

			# Apply normalization and inversion for "lower is better" metrics
			if metric in LOWER_IS_BETTER_METRICS:
				if df[metric].max() == 0 or pd.isna(df[metric].max()):
					df[norm_col] = 0  # Avoid division by zero
				else:
					df[norm_col] = 1 - minmax_scaled_values  # Invert scaling
			else:
				df[norm_col] = minmax_scaled_values  # Normal MinMax scaling

			# Compute Z-score (handle zero variance)
			std_dev = df[norm_col].std()
			if std_dev > 0:
				df[zscore_col] = (df[norm_col] - df[norm_col].mean()) / std_dev
			else:
				df[zscore_col] = np.nan  # Avoid division by zero errors

			# Log normalization details if ranking_config is provided
			if ranking_config is not None:
				for metric_config in ranking_config["metrics"]:
					if metric_config["metric"] == metric:
						metric_config["was_normalized"] = True
						metric_config["was_zscored"] = True
						metric_config["was_inverted"] = metric in LOWER_IS_BETTER_METRICS
						break  # Stop after updating the correct metric

		except ValueError as e:
			console.print(f"[red]Skipping normalization for {metric}: {e}[/red]")
			df[norm_col] = np.nan  # Assign NaN for debugging
			df[zscore_col] = np.nan  # Assign NaN for debugging

			# Log error in ranking_config if provided
			if ranking_config is not None:
				for metric_config in ranking_config["metrics"]:
					if metric_config["metric"] == metric:
						metric_config["ignore_metric"] = True
						metric_config["removal_reason"] = "Normalization Error"
						break

	return df, ranking_config

def calculate_normalized_weighted_scores_by_metric_type(df: pd.DataFrame, metric_weights: dict, metric_type: str, prefix: str = "") -> tuple:
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
	
	Returns:
	--------
	tuple: pd.DataFrame, list
		ranked_comparison_df : pd.DataFrame
			DataFrame containing the comparison results and rankings.
		columns : list
			List of columns used for comparison.
		correlation_rank : float
			Correlation between weighted and summed ranks.
	"""
	if not metric_weights:  # Ensure metrics exist
		console.print(f"[yellow]No metrics provided for {metric_type}. Skipping calculations.[/yellow]")
		return df.copy(), []
	
	# Compute weighted scores
	tqdm.pandas(desc=f"Calculating {metric_type} weighted scores")
	
	# Compute weighted scores
	df[f"{prefix}{metric_type}_weighted_score"] = df.progress_apply(
		lambda row: sum(
			weight * row[f"{prefix}{metric}_normalized"]
			for metric, weight in metric_weights.items()
			if f"{prefix}{metric}_normalized" in df.columns and pd.notna(row[f"{prefix}{metric}_normalized"])
		),
		axis=1
	)

	# Normalize weighted scores
	max_score = df[f"{prefix}{metric_type}_weighted_score"].max()
	if pd.notna(max_score) and max_score > 0:
		df[f"{prefix}{metric_type}_normalized_weighted_score"] = df[f"{prefix}{metric_type}_weighted_score"] / max_score
	else:
		console.print(f"[yellow]Max score is zero or NaN for {metric_type}. Assigning NaN.[/yellow]")
		df[f"{prefix}{metric_type}_normalized_weighted_score"] = np.nan

	# Rank based on weighted scores
	ranked_comparison_df = df.sort_values(
		by=f"{prefix}{metric_type}_normalized_weighted_score", ascending=False
	).reset_index(drop=True)
	ranked_comparison_df[f"{prefix}{metric_type}_normalized_weighted_rank"] = ranked_comparison_df.index + 1

	# Compute summed scores across all metrics
	columns = [f"{prefix}{metric}_normalized" for metric in metric_weights.keys() if f"{prefix}{metric}_normalized" in df.columns]
	ranked_comparison_df[f"{prefix}{metric_type}_summed_score"] = ranked_comparison_df[columns].sum(axis=1)

	# Normalize summed scores
	max_summed_score = ranked_comparison_df[f"{prefix}{metric_type}_summed_score"].max()
	if pd.notna(max_summed_score) and max_summed_score > 0:
		ranked_comparison_df[f"{prefix}{metric_type}_normalized_summed_score"] = ranked_comparison_df[f"{prefix}{metric_type}_summed_score"] / max_summed_score
	else:
		console.print(f"[yellow]Max summed score is zero or NaN for {metric_type}. Assigning NaN.[/yellow]")
		ranked_comparison_df[f"{prefix}{metric_type}_normalized_summed_score"] = np.nan

	# Rank based on summed score
	ranked_comparison_df = ranked_comparison_df.sort_values(
		by=f"{prefix}{metric_type}_normalized_summed_score", ascending=False
	).reset_index(drop=True)
	ranked_comparison_df[f"{prefix}{metric_type}_normalized_summed_rank"] = ranked_comparison_df.index + 1

	# Compute correlation safely
	rank_columns = [
		f"{prefix}{metric_type}_normalized_weighted_rank",
		f"{prefix}{metric_type}_normalized_summed_rank"
	]
	if all(col in ranked_comparison_df.columns for col in rank_columns):
		correlation_rank = ranked_comparison_df[rank_columns].corr().fillna(0).iloc[0, 1]
	else:
		correlation_rank = np.nan

	return ranked_comparison_df, columns, correlation_rank

def normalize_weights_dynamically(metrics: list, weights: dict, results_df: pd.DataFrame, ranking_config: dict, threshold: float = 0.9, shared_weight_factor: float = 0.7, specific_weight_factor: float = 0.3,min_weight: float = 0.05, max_weight: float = 0.5) -> tuple:
	"""
	This function dynamically normalize weights for metrics based on variance, presence, and distribution across shared and specific metrics. It should be run on the normalized values to avoid skewing results. It ensures that metrics are appropriately prioritized, reflecting their relevance and availability, while also maintaining consistency across all metrics. It first checks the variance and presence of each metric. Variance reflects the amount of variation in a metric across the dataset. Metrics with low variance may not provide meaningful distinctions between wavelet configurations, as they exhibit minimal variability. Presence indicates the proportion of data rows where the metric is non-null. Metrics with higher presence values are more widely applicable and thus hold more weight in decision-making. 
	
	The function then identifies shared and specific metrics based on a predefined threshold. Shared metrics are those with a presence above the threshold, while specific metrics have a presence below the threshold. Shared metrics are prioritized over specific metrics, as they are more widely applicable and thus more likely to influence the final ranking. Weights are dynamically adjusted by combining the initial weight with a factor derived from the metric’s variance and presence. Shared metrics receive a higher proportion (e.g., 70%) of the total weight allocation, reflecting their broader relevance, while specific metrics are allocated the remaining proportion (e.g., 30%). The final weights are normalized to sum to 1, ensuring that the total weight is consistent across all metrics. This is crucial for proportional calculations when combining metric scores. Without normalization, metrics with higher raw weights could disproportionately skew results. As a final sanity check, the code loops through the weights and checks if any have been weighted to zero. If that happens it raises an error, as it indicates an unexpected omission or a potential issue with the weighting logic and stops the process. As long as there are no zeros, the function logs the final weights and other relevant information in the ranking configuration for future reference, and then returns the normalized weights and updated ranking configuration.

	Parameters:
	-----------
	existing_metrics : list
		List of metric names to consider.
	weights : dict
		Dictionary of initial weights for each metric.
	results_df : pd.DataFrame
		DataFrame containing the results with metrics.
	ranking_config : dict
		Configuration dictionary to log weight adjustments.
	threshold : float, optional
		Threshold for presence to distinguish shared and specific metrics (default 0.9).
	shared_weight_factor : float, optional
		Factor to prioritize shared metrics (default 0.7).
	specific_weight_factor : float, optional
		Factor to prioritize specific metrics (default 0.3).
	min_weight : float, optional
		Minimum weight assigned to any metric to avoid zeroing out (default 0.05).
	max_weight : float, optional
		Maximum weight allowed for any metric (default 0.5).

	Returns:
	--------
	normalized_weights : dict
		Dictionary of dynamically normalized weights.
	ranking_config : dict
		Updated ranking configuration with normalized weights.
	"""
	# Step 1: Compute Variance and Presence
	metric_variances = results_df[metrics].var()
	metric_presence = results_df[metrics].notna().mean()

	# Step 2: Log Variance and Presence in ranking_config
	for metric in metrics:
		metric = metric.split("_normalized")[0]  # Remove "_normalized" suffix
		for metric_config in ranking_config["metrics"]:
			if metric_config["metric"] == metric:
				metric_config["variance"] = metric_variances.get(metric, None)
				metric_config["presence"] = metric_presence.get(metric, None)

	# Step 3: Adjust Initial Weights Dynamically
	dynamic_adjustments = {
		metric: min(weights.get(metric, min_weight) * max(metric_variances[metric] * metric_presence[metric], min_weight * 10), max_weight)
		for metric in metrics
	}

	# Step 4: Normalize Dynamic Adjustments
	total_adjustment = sum(dynamic_adjustments.values())
	if total_adjustment == 0:
		console.print("[bright_red]All dynamic adjustments are zero. Assigning minimum weights.[/bright_red]")
		normalized_weights = {metric: min_weight for metric in metrics}
	else:
		normalized_weights = {
			metric: min(adjustment / total_adjustment, max_weight)  # Ensure max_weight cap
			for metric, adjustment in dynamic_adjustments.items()
		}

	# Step 5: Split Metrics Based on Shared & Specific Categories
	shared_metrics = [m for m in metrics if metric_presence[m] >= threshold]
	specific_metrics = [m for m in metrics if m not in shared_metrics]

	# If no valid metrics exist, raise an error
	if not shared_metrics and not specific_metrics:
		console.print(metrics)
		raise ValueError("No valid metrics to normalize. Please check metric definitions.")

	# Step 6: Apply Shared/Specific Weight Factors
	shared_total_weight = sum(dynamic_adjustments[m] for m in shared_metrics)
	specific_total_weight = sum(dynamic_adjustments[m] for m in specific_metrics)

	for metric in shared_metrics:
		normalized_weights[metric] = min(
			(dynamic_adjustments[metric] / shared_total_weight) * shared_weight_factor,
			max_weight
		)
		# Log "shared" flag in ranking_config
		metric = metric.split("_normalized")[0]  # Remove "_normalized" suffix
		for metric_config in ranking_config["metrics"]:
			if metric_config["metric"] == metric:
				metric_config["was_shared"] = True

	for metric in specific_metrics:
		normalized_weights[metric] = min(
			(dynamic_adjustments[metric] / specific_total_weight) * specific_weight_factor,
			max_weight
		)
		# Log "specific" flag in ranking_config
		metric = metric.split("_normalized")[0]  # Remove "_normalized" suffix
		for metric_config in ranking_config["metrics"]:
			if metric_config["metric"] == metric:
				metric_config["was_specific"] = True

	# Step 7: Ensure Weights Sum to 1
	total_weight = sum(normalized_weights.values())
	if total_weight == 0:
		console.print("[bright_red]All weights removed! Check your metric definitions.[/bright_red]")
		normalized_weights = {metric: min_weight for metric in metrics}  # Fallback to min_weight
		console.print("[yellow]Fallback: Minimum weights assigned to all metrics.[/yellow]")
	else:
		normalized_weights = {metric: weight / total_weight for metric, weight in normalized_weights.items()}

	# Step 8: Validate Weights and Ensure No Zeros
	for metric in metrics:
		if normalized_weights.get(metric, 0) == 0:
			raise ValueError(
				f"Critical Error: Metric '{metric}' has a weight of zero. "
				"This suggests an unexpected omission or issue in the weighting logic."
			)

	# Step 9: Log Final Weights in ranking_config
	for metric, final_weight in normalized_weights.items():
		metric = metric.split("_normalized")[0]  # Remove "_normalized" suffix
		for metric_config in ranking_config["metrics"]:
			if metric_config["metric"] == metric:
				metric_config["final_weight"] = final_weight
				metric_config["normalized_weight"] = final_weight / sum(normalized_weights.values())  # Normalize to sum to 1

	return normalized_weights, ranking_config

def calculate_dynamically_normalized_weighted_score_by_metric_type(
	normalized_df: pd.DataFrame, updated_metrics: list, ranking_config: dict, weights: dict, 
	prefix: str, metric_type: str, epsilon_threshold: float = 1e-6, penalty_weight: float = 0.05
):
	"""
	Computes dynamically normalized weighted scores, including:
	- A dynamically weighted summed score
	- Weighted penalty for missing metrics
	- Stability-adjusted score
	
	Parameters:
	----------
	normalized_df : pd.DataFrame
		DataFrame containing normalized metric values.
	updated_metrics : list of str
		List of metric names used in the weighted sum.
	ranking_config : dict
		Dictionary containing ranking configuration.
	weights : dict
		Dictionary containing predefined metric weights.
	prefix : str
		Prefix for metric column names.
	metric_type : str
		The type of metric being processed (e.g., "reconstruction" or "signal").
	epsilon_threshold : float, optional
		Small value to prevent division by zero.
	penalty_weight : float, optional
		Weight factor for penalizing missing metrics.
	
	Returns:
	--------
	tuple: (pd.DataFrame, dict)
		ranked_results : DataFrame with computed scores and rankings.
		updated_ranking_config : Updated ranking configuration.
	"""
	
	# Normalize weights dynamically
	normalized_weights, updated_ranking_config = normalize_weights_dynamically(updated_metrics, weights, normalized_df, ranking_config)
	
	# Create a lookup dictionary for ignored metrics
	metric_lookup = {m["metric"]: m.get("ignore_metric", False) for m in updated_ranking_config["metrics"]}

	# Compute weighted penalty for missing and ignored metrics
	normalized_df[f'{prefix}{metric_type}_missing_metrics_count'] = normalized_df.apply(
		lambda row: sum(
			normalized_weights.get(metric, 0) * (1 if pd.isna(row[f"{prefix}{metric}_normalized"]) else 0.5)
			for metric in updated_metrics
			if f"{prefix}{metric}_normalized" in normalized_df.columns
			and (pd.isna(row[f"{prefix}{metric}_normalized"]) or metric_lookup.get(metric.replace("_normalized", ""), False))
		),
		axis=1
	)

	# Compute dynamically normalized weighted score
	tqdm.pandas(desc=f"Calculating {metric_type} dynamic scores")
	ranking_config["penalty_weight"] = penalty_weight
	normalized_df[f"{prefix}{metric_type}_dynamically_weighted_score"] = normalized_df.progress_apply(
		lambda row: (
			sum(
				normalized_weights[metric] * row[f"{prefix}{metric}_normalized"]
				for metric in updated_metrics
				if pd.notna(row[f"{prefix}{metric}_normalized"])
			) / max(sum(normalized_weights[metric] for metric in updated_metrics if pd.notna(row[f"{prefix}{metric}_normalized"])), epsilon_threshold)
			- penalty_weight * row[f"{prefix}{metric_type}_missing_metrics_count"]  # Use precomputed penalty
		),
		axis=1
	)

	# Compute dynamically normalized z-score weighted score
	zscore_metrics = [metric.split("_normalized")[0] for metric in updated_metrics]
	normalized_df[f"{prefix}{metric_type}_dynamically_weighted_score_zscore"] = normalized_df.apply(
		lambda row: (
			sum(
				normalized_weights[metric + "_normalized"] * row[f"{prefix}{metric}_zscore"]
				for metric in zscore_metrics
				if pd.notna(row[f"{prefix}{metric}_zscore"])
			) / max(sum(normalized_weights[metric + "_normalized"] for metric in zscore_metrics if pd.notna(row[f"{prefix}{metric}_zscore"])), epsilon_threshold)
		),
		axis=1
	)

	# Normalize the dynamically weighted scores
	max_weighted_score = normalized_df[f"{prefix}{metric_type}_dynamically_weighted_score"].max()
	if max_weighted_score > 0:
		normalized_df[f"{prefix}{metric_type}_normalized_dynamically_weighted_score"] /= max_weighted_score
	else:
		console.print("[yellow]Max dynamically weighted score is zero! Assigning equal scores.[/yellow]")
		normalized_df[f"{prefix}{metric_type}_normalized_dynamically_weighted_score"] = 1 / len(updated_metrics)  # Assign equal importance

	# Normalize the dynamically weighted z-score scores
	max_zscore_weighted_score = normalized_df[f"{prefix}{metric_type}_dynamically_weighted_score_zscore"].max()
	if max_zscore_weighted_score > 0:
		normalized_df[f"{prefix}{metric_type}_normalized_dynamically_weighted_score_zscore"] /= max_zscore_weighted_score
	else:
		console.print("[yellow]Max dynamically weighted z-score score is zero! Assigning equal scores.[/yellow]")
		normalized_df[f"{prefix}{metric_type}_normalized_dynamically_weighted_score_zscore"] = 1 / len(updated_metrics)  # Assign equal importance

	# Final stability-adjusted score 
	normalized_df[f"{prefix}{metric_type}_final_normalized_dynamic_score"] = (
	(normalized_df[f"{prefix}{metric_type}_normalized_dynamically_weighted_score"] + 
		normalized_df[f"{prefix}{metric_type}_normalized_dynamically_weighted_score_zscore"]) / 2
	)

	# Rank results by final dynamic score
	ranked_results = normalized_df.sort_values(
	by=f"{prefix}{metric_type}_final_normalized_dynamic_score", ascending=False
	).reset_index(drop=True)
	ranked_results[f"{prefix}{metric_type}_dynamic_rank"] = ranked_results.index + 1


	# Rank results by final dynamic score
	ranked_results = normalized_df.sort_values(
		by=f"{prefix}{metric_type}_dynamic_score", ascending=False
	).reset_index(drop=True)
	ranked_results[f"{prefix}{metric_type}_dynamic_rank"] = ranked_results.index + 1

	# Compute dynamically weighted summed score
	normalized_df[f"{prefix}{metric_type}_dynamically_normalized_weighted_summed_score"] = normalized_df.apply(
		lambda row: sum(
			normalized_weights.get(metric, 0) * row[f"{prefix}{metric}_normalized"]
			for metric in updated_metrics
			if f"{prefix}{metric}_normalized" in normalized_df.columns and pd.notna(row[f"{prefix}{metric}_normalized"])
		),
		axis=1
	)

	# Normalize summed score
	max_summed_score = normalized_df[f"{prefix}{metric_type}_dynamically_normalized_weighted_summed_score"].max()
	if max_summed_score > 0:
		normalized_df[f"{prefix}{metric_type}_dynamically_normalized_weighted_summed_score"] /= max_summed_score
	else:
		console.print("[yellow]Max summed normalized score is zero! Assigning equal scores.[/yellow]")
		normalized_df[f"{prefix}{metric_type}_dynamically_normalized_weighted_summed_score"] = 1 / len(updated_metrics)  # Assign equal importance

		# Compute normalized diff for summed score
	normalized_df[f"{prefix}{metric_type}_normalized_summed_diff"] = (
		(normalized_df[f"{prefix}{metric_type}_dynamically_normalized_weighted_summed_score"] - 
		 normalized_df[f"{prefix}{metric_type}_dynamically_normalized_weighted_score_zscore"]).abs()
		/ (normalized_df[f"{prefix}{metric_type}_dynamically_normalized_weighted_summed_score"] + 
		   normalized_df[f"{prefix}{metric_type}_dynamically_normalized_weighted_score_zscore"]).abs()
	).abs()

	# Final stability-adjusted summed score
	normalized_df[f"{prefix}{metric_type}_dynamic_summed_score"] = (
		normalized_df[f"{prefix}{metric_type}_dynamically_normalized_weighted_summed_score"]
		- penalty_weight * normalized_df[f"{prefix}{metric_type}_normalized_summed_diff"]
	)

	# Rank results by dynamically weighted summed score
	ranked_results = ranked_results.sort_values(
		by=f"{prefix}{metric_type}_dynamic_weighted_summed_score", ascending=False
	).reset_index(drop=True)
	ranked_results[f"{prefix}{metric_type}_dynamic_summed_rank"] = ranked_results.index + 1

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
	for score_type in ["weighted", "summed", "dynamic"]:
		for metric_type in ["reconstruction", "signal"]:
			score_col = f"{prefix}{metric_type}_{score_type}_score"
			if score_col in ranked_results.columns:
				ranking_config[f"{metric_type}_{score_type}_max"] = ranked_results[score_col].max()
				ranking_config[f"{metric_type}_{score_type}_min"] = ranked_results[score_col].min()
				ranking_config[f"{metric_type}_{score_type}_mean"] = ranked_results[score_col].mean()
				ranking_config[f"{metric_type}_{score_type}_std"] = ranked_results[score_col].std()

	return ranking_config

def calculate_rank_stability(df: pd.DataFrame, rank_columns: list, prefix: str, weight_factor: float = 0.5) -> pd.DataFrame:
	"""
	Calculate a stability metric for wavelet rankings based on multiple ranking columns.
	
	Parameters:
	-----------
	df : pd.DataFrame
		DataFrame containing rank columns to evaluate.
	rank_columns : list of str
		Columns representing ranks to compare for stability.
	prefix : str
		Prefix for column names (e.g., "combined_" for merged datasets).
	weight_factor : float, optional
		Weight factor for the stability metric (default is 0.5).
		
	Returns:
	--------
	pd.DataFrame
		DataFrame with additional stability metrics and final stability rank.
	"""
	# Calculate rank variability (standard deviation across ranking columns)
	df[f'{prefix}rank_variability'] = df[rank_columns].std(axis=1)

	# Normalize stability score between 0 and 1
	max_rank = df[rank_columns].max().max()
	df[f'{prefix}normalized_rank_stability'] = 1 - (df[f'{prefix}rank_variability'] / max_rank)

	# Compute weighted mean rank (stability weighted)
	df[f'{prefix}average_rank'] = df[rank_columns].mean(axis=1)
	df[f'{prefix}weighted_average_rank'] = (
		df[f'{prefix}average_rank'] * (1 - weight_factor) + df[f'{prefix}normalized_rank_stability'] * weight_factor
	)

	# Compute harmonic mean of ranks (to penalize large variations)
	df[f'{prefix}harmonic_average_rank'] = len(rank_columns) / np.sum(1 / df[rank_columns], axis=1)

	# Compute row-wise rank correlation across rank columns
	df[f'{prefix}rank_correlation'] = df[rank_columns].corrwith(df[rank_columns].mean(axis=1), axis=1)

	# Normalize correlation so it's within [0,1] range
	df[f'{prefix}normalized_rank_correlation'] = (df[f'{prefix}rank_correlation'] - df[f'{prefix}rank_correlation'].min()) / (
		df[f'{prefix}rank_correlation'].max() - df[f'{prefix}rank_correlation'].min() + 1e-6
	)

	# Compute final stability score combining rank stability and correlation
	df[f'{prefix}stability_score_weighted'] = (
		df[f'{prefix}weighted_average_rank'] * (1 - weight_factor) + df[f'{prefix}normalized_rank_correlation'] * weight_factor
	)

	# Sort by stability score (lower is better)
	df = df.sort_values(by=f'{prefix}stability_score_weighted', ascending=True).reset_index(drop=True)

	# Assign final stability ranking
	df[f'{prefix}stability_rank'] = df.index + 1
	
	return df

def calculate_wavelet_family_scores(df: pd.DataFrame, prefix: str, rank_bins:list=[0, 10, 20, 50, 100, None],):
	"""
	Computes wavelet scores for either individual volumes or combined titles.

	Parameters:
	- df: DataFrame containing the wavelet data.
	- level: 'individual' for individual volume analysis, 'combined' for title-level analysis.
	- rank_bins: List of bin edges for rank binning.
	- weights: Dictionary of weights for the composite score. Example:
		{
			'composite_score': 0.3,
			'rank_stability': 0.25,
			'mean_rank': 0.15,
			'total_count': 0.1,
			'global_proportion': 0.1,
			'htid_proportion': 0.1
		}
	
	Returns:
	- Processed DataFrame with calculated scores and rankings.
	"""
	# Extract wavelet families
	df['wavelet_family'] = df['wavelet'].str.extract(r'([a-zA-Z]+)').fillna('Unknown')

	# Normalize rank and stability
	df[f'{prefix}normalized_weighted_mean_rank'] = df[f'{prefix}weighted_mean_rank'] / df[f'{prefix}weighted_mean_rank'].max()
	df[f'{prefix}normalized_rank_stability'] = 1 - df[f'{prefix}rank_stability']

	# Define weights for rank and stability
	alpha = 0.5  # Weight for rank
	beta = 0.5   # Weight for stability

	# Compute composite score
	df[f'{prefix}composite_score'] = (
		alpha * df[f'{prefix}normalized_weighted_mean_rank'] + beta * df[f'{prefix}normalized_rank_stability']
	)
	
	# Assign rank bins
	if rank_bins[-1] is None:  # Replace None with max rank
		rank_bins[-1] = df[f'{prefix}weighted_mean_rank'].max()

	df[f'{prefix}rank_bin'] = pd.cut(
		df[f'{prefix}weighted_mean_rank'],
		bins=rank_bins,
		labels=[f"Top {int(bin_edge)}" if bin_edge != rank_bins[-1] else "Beyond 100" for bin_edge in rank_bins[1:]]
	)
	# Add rank bin summaries
	rank_bin_summary = df.groupby(['wavelet_family', f'{prefix}rank_bin']).agg(
		binned_count=(f'{prefix}weighted_mean_rank', 'count'),
		binned_unique_htid=('htid', 'nunique'),  # Count of unique volumes (htid)
		binned_mean_rank_stability=(f'{prefix}rank_stability', 'mean'),  # Mean rank stability
		binned_std_rank_stability=(f'{prefix}rank_stability', 'std')  # Standard deviation of rank stability
	).reset_index()

	# Add proportions
	rank_bin_summary[f'{prefix}global_proportion'] = rank_bin_summary['binned_count'] / rank_bin_summary.groupby(f'{prefix}rank_bin')['binned_count'].transform('sum')
	rank_bin_summary[f'{prefix}htid_proportion'] = rank_bin_summary['binned_unique_htid'] / rank_bin_summary.groupby(f'{prefix}rank_bin')['binned_unique_htid'].transform('sum')

	top10_bins = rank_bin_summary[rank_bin_summary[f'{prefix}rank_bin'] == 'Top 10'].sort_values(by=[f'{prefix}global_proportion', f'{prefix}htid_proportion', 'binned_mean_rank_stability', 'binned_std_rank_stability', 'binned_count', 'binned_unique_htid'], ascending=[False, False, False, True, False, False])

	all_bins = rank_bin_summary.sort_values(by=[f'{prefix}global_proportion', f'{prefix}htid_proportion', 'binned_mean_rank_stability', 'binned_std_rank_stability', 'binned_count', 'binned_unique_htid'], ascending=[False, False, False, True, False, False])


	# Aggregate metrics based on the level
	wavelet_summary = df.groupby('wavelet_family').agg(
		wavelet_family_mean_composite_score=(f'{prefix}composite_score', 'mean'),
		wavelet_family_mean_rank_stability=(f'{prefix}rank_stability', 'mean'),
		wavelet_family_std_rank_stability=(f'{prefix}rank_stability', 'std'),
		wavelet_family_mean_rank=(f'{prefix}weighted_mean_rank', 'mean'),
		wavelet_family_total_count=('htid', 'count')
	).reset_index()

	top10_wavelet_summary = wavelet_summary.merge(top10_bins, on='wavelet_family', how='left')
	top10_numeric_cols = top10_wavelet_summary.select_dtypes(include="number").columns.tolist()
	top10_wavelet_summary[top10_numeric_cols] = top10_wavelet_summary[top10_numeric_cols].fillna(0)	
	all_wavelet_summary = wavelet_summary.merge(all_bins, on='wavelet_family', how='left')	
	all_numeric_cols = all_wavelet_summary.select_dtypes(include="number").columns.tolist()
	all_wavelet_summary[all_numeric_cols] = all_wavelet_summary[all_numeric_cols].fillna(0)

	# Normalize all metrics
	for col in ['wavelet_family_mean_composite_score', 'wavelet_family_mean_rank_stability', 'wavelet_family_mean_rank', 'wavelet_family_total_count', f'{prefix}global_proportion', f'{prefix}htid_proportion']:
		top10_wavelet_summary[f'{prefix}normalized_{col}'] = top10_wavelet_summary[col] / top10_wavelet_summary[col].max() if top10_wavelet_summary[col].max() != 0 else 0
		all_wavelet_summary[f'{prefix}normalized_{col}'] = all_wavelet_summary[col] / all_wavelet_summary[col].max() if all_wavelet_summary[col].max() != 0 else 0

	# Adjust ranking: Lower values for mean rank and composite score are better, so invert them
	top10_wavelet_summary[f'{prefix}inv_normalized_wavelet_family_mean_composite_score'] = 1 - top10_wavelet_summary[f'{prefix}normalized_wavelet_family_mean_composite_score']
	top10_wavelet_summary[f'{prefix}inv_normalized_wavelet_family_mean_rank'] = 1 - top10_wavelet_summary[f'{prefix}normalized_wavelet_family_mean_rank']
	all_wavelet_summary[f'{prefix}inv_normalized_wavelet_family_mean_composite_score'] = 1 - all_wavelet_summary[f'{prefix}normalized_wavelet_family_mean_composite_score']
	all_wavelet_summary[f'{prefix}inv_normalized_wavelet_family_mean_rank'] = 1 - all_wavelet_summary[f'{prefix}normalized_wavelet_family_mean_rank']

	# Compute final composite score with corrected weighting
	top10_wavelet_summary[f'{prefix}wavelet_composite_score'] = (
		0.3 * top10_wavelet_summary[f'{prefix}inv_normalized_wavelet_family_mean_composite_score'] +  # Lower is better
		0.25 * top10_wavelet_summary[f'{prefix}normalized_wavelet_family_mean_rank_stability'] +  # Higher is better
		0.15 * top10_wavelet_summary[f'{prefix}inv_normalized_wavelet_family_mean_rank'] +  # Lower is better
		0.1 * top10_wavelet_summary[f'{prefix}normalized_wavelet_family_total_count'] +  # Higher is better
		0.1 * top10_wavelet_summary[f'{prefix}normalized_global_proportion'] +  # Higher is better
		0.1 * top10_wavelet_summary[f'{prefix}normalized_htid_proportion']  # Higher is better
	)

	# Compute final composite score with corrected weighting
	all_wavelet_summary[f'{prefix}wavelet_composite_score'] = (
		0.3 * all_wavelet_summary[f'{prefix}inv_normalized_wavelet_family_mean_composite_score'] +  # Lower is better
		0.25 * all_wavelet_summary[f'{prefix}normalized_wavelet_family_mean_rank_stability'] +  # Higher is better
		0.15 * all_wavelet_summary[f'{prefix}inv_normalized_wavelet_family_mean_rank'] +  # Lower is better
		0.1 * all_wavelet_summary[f'{prefix}normalized_wavelet_family_total_count'] +  # Higher is better
		0.1 * all_wavelet_summary[f'{prefix}normalized_global_proportion'] +  # Higher is better
		0.1 * all_wavelet_summary[f'{prefix}normalized_htid_proportion']  # Higher is better
	)

	# Sort by final score
	top10_wavelet_summary = top10_wavelet_summary.sort_values(by=['rank_bin', f'wavelet_composite_score'], ascending=[True, False])
	all_wavelet_summary = all_wavelet_summary.sort_values(by=['rank_bin', f'wavelet_composite_score'], ascending=[True, False])

	# Select the best wavelet
	top10_wavelet_family = top10_wavelet_summary.iloc[0].wavelet_family
	all_wavelet_family = all_wavelet_summary.iloc[0].wavelet_family

	if top10_wavelet_family != all_wavelet_family:
		console.print(f"Best wavelet family (Top 10): {top10_wavelet_family}", style="bright_magenta")
		console.print(top10_wavelet_summary[['wavelet_family', f'{prefix}wavelet_composite_score', "wavelet_family_mean_rank"]], style="bright_magenta")
		console.print(f"Best wavelet family (All): {all_wavelet_family}", style="bright_magenta")
		console.print(all_wavelet_summary[['wavelet_family', f'{prefix}wavelet_composite_score', "wavelet_family_mean_rank"]], style="bright_magenta")
		# Ask user which wavelet family to use
		final_wavelet_family = console.input("Which wavelet family would you like to use? (Top 10/All): ")
		if final_wavelet_family.lower() == 'top 10':
			console.print(f"Selected Top 10", style="bright_green")
			final_wavelet_summary = top10_wavelet_summary
		else:
			console.print(f"Selected All", style="bright_green")
			final_wavelet_summary = all_wavelet_summary
	else:
		console.print(f"Best wavelet family across both Top 10 and All: {top10_wavelet_family}", style="bright_magenta")
		final_wavelet_summary = all_wavelet_summary
		

	final_df = df.merge(final_wavelet_summary, on=['wavelet_family', f'{prefix}rank_bin'], how='left')
	final_df = final_df.sort_values(by=[f'{prefix}weighted_mean_rank', f'{prefix}wavelet_composite_score'], ascending=[True, False])
	final_df[f"{prefix}final_wavelet_rank"] = final_df.index + 1
	return final_df

def select_top_ranked_results(ranked_results: pd.DataFrame, prefix: str, percentage_of_results: float = 0.1) -> pd.DataFrame:
	"""
	Selects the top-ranked wavelet configurations dynamically based on the final score.
	Ensures that the best-performing configurations are chosen within the top N% of results,
	while also selecting the best configurations for each wavelet type.

	Parameters:
	----------
	ranked_results : pd.DataFrame
		DataFrame containing ranked results with computed scores.
	prefix : str
		Prefix for column names (e.g., "combined_" for merged datasets).
	is_combined : bool
		Whether results are combined across multiple wavelet types.
	percentage_of_results : float, optional
		Percentage of results to retain (default is 10%).

	Returns:
	--------
	pd.DataFrame
		DataFrame with the final top-ranked wavelet configurations.
	"""

	# Dynamically select the top N% of ranked results
	num_top_results = max(1, int(len(ranked_results) * percentage_of_results))  # Ensure at least one result
	top_ranked_results = ranked_results.head(num_top_results)

	# Select top configurations by wavelet, signal type, and wavelet type
	grouping_cols = ['wavelet_type', 'wavelet'] if len(prefix) > 0 else ['wavelet']
	grouped = ranked_results.groupby(grouping_cols)

	subset_ranked_results = grouped.apply(
		lambda group: group.loc[group[f"{prefix}final_wavelet_rank"].idxmax()]
	).reset_index(drop=True)

	# Identify columns to drop to avoid duplicate selection
	drop_cols = ['signal_type', 'wavelet_type', 'wavelet', 'wavelet_mode', 'wavelet_level']
	drop_cols = [col for col in drop_cols if col in subset_ranked_results.columns and col in top_ranked_results.columns]

	# Merge and deduplicate top results with best group selections
	final_ranked_results = pd.concat([top_ranked_results, subset_ranked_results], ignore_index=True)
	final_ranked_results = final_ranked_results.drop_duplicates(subset=drop_cols, keep='first')

	# Sort final results by final score
	final_ranked_results = final_ranked_results.sort_values(
		by=f"{prefix}weighted_mean_rank", ascending=False
	).reset_index(drop=True)

	# Assign final ranking
	final_ranked_results[f"{prefix}top_wavelet_rank"] = final_ranked_results.index + 1

	return final_ranked_results

def determine_best_wavelet_representation(
	results_df: pd.DataFrame, signal_type: str, original_signal_metrics_df: pd.DataFrame, prefix: str = "", epsilon_threshold: float = 1e-6, penalty_weight: float = 0.05, percentage_of_results: float = 0.1, ignore_low_variance: bool = False
) -> tuple:
	# Generate ranking configuration
	ranking_config = generate_ranking_config(signal_type, prefix)

	if len(prefix) == 0:
		initial_preprocessed_results_df = preprocess_reconstructed_metrics(original_signal_metrics_df, results_df)
		preprocessed_results_df, ranking_config = preprocess_signal_metrics(initial_preprocessed_results_df, ranking_config, ignore_low_variance, epsilon_threshold)

	else:
		preprocessed_results_df, ranking_config = preprocess_signal_metrics(results_df, ranking_config, ignore_low_variance, epsilon_threshold)
	
	metrics = [
		metric_config["metric"]
		for metric_config in ranking_config["metrics"]
		if not metric_config.get("ignore_metric", False)  # Exclude ignored metrics
		and metric_config["metric"] in preprocessed_results_df.columns  # Ensure existence in `results_df`
	]
	
	# Normalize and Z-score metrics
	normalized_results_df, ranking_config = normalize_and_zscore(preprocessed_results_df, metrics, ranking_config, prefix)

	# Calculate summed normalized scores
	normalized_results_df["across_all_metrics_summed_normalized_score"] = normalized_results_df.filter(like="_normalized").sum(axis=1)

	# Calculate weighted normalized scores
	partial_weighted_scored_df, reconstruction_columns, reconstruction_rank_correlation = calculate_normalized_weighted_scores_by_metric_type(normalized_results_df, RECONSTRUCTION_METRIC_WEIGHTS, "reconstruction")
	full_weighted_scored_df, signal_columns, signal_rank_correlation = calculate_normalized_weighted_scores_by_metric_type(partial_weighted_scored_df, SIGNAL_METRIC_WEIGHTS, "signal")

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


	# Calculate rank stability
	rank_cols = [f"{prefix}reconstruction_normalized_weighted_rank", f"{prefix}reconstruction_dynamic_rank", f"{prefix}signal_normalized_weighted_rank", f"{prefix}signal_dynamic_rank"] 
	stable_ranked_results = calculate_rank_stability(final_ranked_results, rank_cols, prefix)
	total_ranked_results = calculate_wavelet_family_scores(stable_ranked_results, prefix)
	
	top_ranked_results = select_top_ranked_results(total_ranked_results, prefix, percentage_of_results)

	return top_ranked_results, total_ranked_results, final_ranking_config