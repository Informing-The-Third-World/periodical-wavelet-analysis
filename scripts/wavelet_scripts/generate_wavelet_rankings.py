# Standard library imports
import os
import sys
import warnings
from typing import Any, Tuple, Union

# Third-party imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from rich.console import Console
import pywt
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.stats import wasserstein_distance
from difflib import SequenceMatcher
from fastdtw import fastdtw
from minineedle import needle, core

# Local application imports
sys.path.append("../..")
from scripts.utils import read_csv_file, get_data_directory_path

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

## WAVELET RANKING AND COMPARISON FUNCTIONS
def normalize_weights_dynamically(existing_metrics: list, weights: dict, results_df: pd.DataFrame, ranking_config: dict, threshold: float = 0.9, shared_weight_factor: float = 0.7, specific_weight_factor: float = 0.3,min_weight: float = 0.05, max_weight: float = 0.5) -> tuple:
	"""
	This function dynamically normalize weights for metrics based on variance, presence, and distribution across shared and specific metrics. It ensures that metrics are appropriately prioritized, reflecting their relevance and availability, while also maintaining consistency across all metrics. It first checks the variance and presence of each metric. Variance reflects the amount of variation in a metric across the dataset. Metrics with low variance may not provide meaningful distinctions between wavelet configurations, as they exhibit minimal variability. Presence indicates the proportion of data rows where the metric is non-null. Metrics with higher presence values are more widely applicable and thus hold more weight in decision-making. 
	
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
	metrics = [metric for metric in existing_metrics if metric in results_df.columns]

	# Step 1: Compute Variance and Presence
	metric_variances = results_df[metrics].var()
	metric_presence = results_df[metrics].notna().mean()

	# Step 2: Log Variance and Presence in ranking_config
	for metric in metrics:
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
		for metric_config in ranking_config["metrics"]:
			if metric_config["metric"] == metric:
				metric_config["was_shared"] = True

	for metric in specific_metrics:
		normalized_weights[metric] = min(
			(dynamic_adjustments[metric] / specific_total_weight) * specific_weight_factor,
			max_weight
		)
		# Log "specific" flag in ranking_config
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
		for metric_config in ranking_config["metrics"]:
			if metric_config["metric"] == metric:
				metric_config["final_weight"] = final_weight
				metric_config["normalized_weight"] = final_weight / sum(normalized_weights.values())  # Normalize to sum to 1

	return normalized_weights, ranking_config

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

def compare_original_reconstructed_metrics(
	original_df: pd.DataFrame, reconstructed_df: pd.DataFrame
) -> tuple:
	"""
	Compares the outputs from the calculate_signal_metrics function for the original signal versus all reconstructed wavelets. It assumes that the comparison will be within signal type (so not across raw and smoothed results). It also assumes that the original_df contains only one row of data. The function compares the metrics for each wavelet and returns a DataFrame with the comparison results. It uses a combination of sequence alignment, DTW, and other distance measures to compare the metrics. The function also normalizes the scores and computes a final reconstruction score for each wavelet.

	Parameters:
	-----------
	original_df : pd.DataFrame
		DataFrame containing the original signal metrics.
	reconstructed_df : pd.DataFrame
		DataFrame containing the reconstructed signal metrics.
	Returns:
	--------
	ranked_comparison_df : pd.DataFrame
		DataFrame containing the comparison results and reconstruction scores.
	ranking_config : dict
		Dictionary containing the ranking configuration.
	"""

	if original_df.empty or reconstructed_df.empty:
		console.print("[yellow]One or both DataFrames are empty. Returning empty results.[/yellow]")
		return pd.DataFrame()

	if len(original_df) != 1:
		raise ValueError("original_df must contain exactly one row for comparison.")

	ranking_config = {
        "signal_type": original_df["signal_type"].iloc[0],
        "is_reconstruction_comparison": True,
        "metrics": []
    }
	comparison_rows = []  # To store row-by-row comparison results
	metrics = set(original_df.columns).intersection(set(reconstructed_df.columns)) - {"signal_type"}

	cols_to_add = ['wavelet', 'wavelet_level','wavelet_mode']
	cols_to_add = [col for col in cols_to_add if col in reconstructed_df.columns]

	for idx, reconstructed_row in tqdm(reconstructed_df.iterrows(), total=reconstructed_df.shape[0], desc="Comparing metrics"):
		comparison = {"row_index": idx}  # Track the row index for reference
		comparison.update({col: reconstructed_row[col] for col in cols_to_add})
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
						comparison[f"{metric}_matcher_alignment_score"] = align_arrays(original_array, reconstructed_array)
						comparison[f"{metric}_global_alignment_score"] = run_global_sequence_alignment(original_array, reconstructed_array)
					elif metric in {"positive_frequencies", "positive_amplitudes"}:
						comparison[f"{metric}_dtw"] = compare_distributions(original_array, reconstructed_array, method="dtw")
						if len(original_array) == len(reconstructed_array):
							comparison[f"{metric}_euclidean"] = compare_distributions(original_array, reconstructed_array, method="euclidean")
						comparison[f"{metric}_wasserstein"] = compare_distributions(original_array, reconstructed_array, method="wasserstein")
					elif metric == "relative_prominences":
						avg_diff, total_diff, wasserstein = compare_prominences_distribution(original_array, reconstructed_array)
						comparison[f"{metric}_avg_diff"] = avg_diff
						comparison[f"{metric}_total_diff"] = total_diff
						comparison[f"{metric}_wasserstein"] = wasserstein
				else:
					try:
						comparison[f"{metric}_diff"] = abs(float(original_df[metric].iloc[0]) - float(reconstructed_row[metric]))
					except Exception as e:
						console.print(f"[red]Error computing difference for metric '{metric}': {e}[/red]")
						comparison[f"{metric}_diff"] = np.nan

		comparison_rows.append(comparison)

	# Create DataFrame from all rows
	comparison_df = pd.DataFrame(comparison_rows)
	normalized_df = comparison_df.copy()
	scaler = MinMaxScaler(feature_range=(0, 1))  # Ensure MinMaxScaler is always bounded within [0,1]

	lower_is_better_metrics = [
		col for col in comparison_df.columns if "diff" in col or col in [
			"wasserstein", "dtw", "euclidean", "avg_diff", "total_diff"
		]
	]

	for col in comparison_df.columns:

		if col.endswith("_normalized") or col == "row_index":
			continue  # Skip already normalized columns or non-metric columns
		max_value = comparison_df[col].max()
		min_value = comparison_df[col].min()
		console.print(f"Processing column for normalization: {col}, Initial Max: {max_value}, Initial Min: {min_value}")
		# Normalize "lower is better" metrics by inverting their values
		if col in lower_is_better_metrics:
	
			if pd.isna(max_value) or max_value == 0:
				normalized_df[col + "_normalized"] = 0  # Avoid division by zero
			else:
				normalized_df[col + "_normalized"] = 1 - (comparison_df[col] / max_value)

		# Normalize "higher is better" metrics using MinMaxScaler
		elif col not in cols_to_add + ["row_index"]:
			try:
				normalized_df[col + "_normalized"] = scaler.fit_transform(
					comparison_df[col].values.reshape(-1, 1)
				).flatten()
			except ValueError as e:
				console.print(f"[red]Skipping normalization for {col}: {e}[/red]")
				normalized_df[col + "_normalized"] = np.nan  # Assign NaN for debugging
		norm_max_value = normalized_df[col + "_normalized"].max()
		norm_min_value = normalized_df[col + "_normalized"].min()
		console.print(f"Normalized column: {col}, Normalized Max: {norm_max_value}, Normalized Min: {norm_min_value}")

	# Define weights for all metric types
	metric_weights = {
		"diff": 0.3,              # Medium weight for differences
		"matcher_alignment_score": 0.6,  # High weight for sequence alignment
		"global_alignment_score": 0.6,  # High weight for global alignment
		"dtw": 0.5,               # Medium weight for DTW distances
		"euclidean": 0.4,         # Medium-low weight for Euclidean distances
		"wasserstein": 0.4,       # Medium-low weight for Wasserstein distances
		"avg_diff": 0.3,          # Low weight for average difference
		"total_diff": 0.2,        # Low weight for total difference
	}

	# Compute reconstruction scores
	total_scores_weighted = []
	for _, row in normalized_df.iterrows():
		weighted_scores = []
		for metric_type, weight in metric_weights.items():
			metric_columns = [col for col in normalized_df.columns if col.endswith(metric_type + "_normalized")]
			if metric_columns:
				metric_score = row[metric_columns].mean()  # Average across related normalized metrics
				weighted_scores.append(weight * metric_score)

		total_scores_weighted.append(sum(weighted_scores))

	normalized_df["reconstruction_score_weighted"] = total_scores_weighted

	# Compute **simple summation reconstruction score**
	normalized_columns = [col for col in normalized_df.columns if col.endswith("_normalized")]
	normalized_df["reconstruction_score_sum"] = normalized_df[normalized_columns].sum(axis=1)

	# Correlation Analysis
	correlation_matrix = normalized_df[normalized_columns + ["reconstruction_score_weighted", "reconstruction_score_sum"]].corr()
	reconstruction_corr = correlation_matrix["reconstruction_score_weighted"].sort_values(ascending=False)
	console.print(f"Weighted Reconstruction Score Correlation:\n{reconstruction_corr}")

	reconstruction_corr_sum = correlation_matrix["reconstruction_score_sum"].sort_values(ascending=False)
	console.print(f"Summed Reconstruction Score Correlation:\n{reconstruction_corr_sum}")

	# Sort results
	ranked_comparison_df = normalized_df.sort_values(by="reconstruction_score_weighted", ascending=False).reset_index(drop=True)
	ranked_comparison_df["rank_weighted"] = ranked_comparison_df.index + 1  # Rank by weighted method
	ranked_comparison_df = ranked_comparison_df.sort_values(by="reconstruction_score_sum", ascending=False).reset_index(drop=True)
	ranked_comparison_df["rank_sum"] = ranked_comparison_df.index + 1  # Rank by summed method

	for metric in metrics:
		ranking_config["metrics"].append({
			"metric": metric,
			"weight": next((metric_weights[mt] for mt in metric_weights if metric.endswith(mt)), None),
			"was_normalized": metric + "_normalized" in normalized_df.columns,
			"was_inverted": metric in lower_is_better_metrics
		})

	ranking_config["reconstruction_score_weighted_max"] = normalized_df["reconstruction_score_weighted"].max()
	ranking_config["reconstruction_score_weighted_min"] = normalized_df["reconstruction_score_weighted"].min()
	ranking_config["reconstruction_score_weighted_mean"] = normalized_df["reconstruction_score_weighted"].mean()
	ranking_config["reconstruction_score_weighted_std"] = normalized_df["reconstruction_score_weighted"].std()
	ranking_config["reconstruction_score_sum_max"] = normalized_df["reconstruction_score_sum"].max()
	ranking_config["reconstruction_score_sum_min"] = normalized_df["reconstruction_score_sum"].min()
	ranking_config["reconstruction_score_sum_mean"] = normalized_df["reconstruction_score_sum"].mean()
	ranking_config["reconstruction_score_sum_std"] = normalized_df["reconstruction_score_sum"].std()

	return ranked_comparison_df, ranking_config

def determine_best_wavelet_representation(
	results_df: pd.DataFrame, signal_type: str, original_signal_metrics_df: pd.DataFrame, weights: dict = None, is_combined: bool = False, epsilon_threshold: float = 1e-6, penalty_weight: float = 0.05, percentage_of_results: float = 0.1, ignore_low_variance: bool = False
) -> tuple:
	"""
	This function determines the best wavelet representation by normalizing, combining scores, and ranking based on provided metrics. It starts by using either the passed weights or the existing default ones. Then it creates a ranking_config dictionary to log the decision-making process, including initial weights, transformations, and exclusions of metrics. Next we start checking for zero or near-zero variance metrics because they do not provide meaningful information for distinguishing between wavelet configurations. Any excluded metric is logged in the ranking_config with it's mean and standard deviation. After that step, we check if we need to log-transform negative values in `wavelet_energy_entropy`, which can distort normalization and weight calculations. The log transformation compresses large ranges while handling negatives, stabilizing the metric for comparison. This step assumes the metric’s distribution benefits from log-scaling to better represent its variation. Finally, we handle complex-valued metrics by taking the absolute value. Complex numbers might arise in certain wavelet calculations (e.g., from Fourier-based transforms). Magnitudes retain the essential information while making the data compatible with subsequent operations. 
	
	Our next step is to normalize these metrics using a RobustScaler. Unlike MinMaxScaler, which can be heavily influenced by extreme outliers, RobustScaler normalizes data based on the interquartile range (IQR), making it more robust for datasets with skewed distributions. We also create a copy (normalized_df) of the original results_df, as well as check if the analysis is within within a single wavelet transform type (e.g., DWT, CWT, SWT) or across all types. This affects how metric columns are prefixed (e.g., combined_). We then start normalizing the metrics. Any skipped metrics are logged in the ranking_config. We also invert metrics where lower values are better (e.g., MSE, entropy) to ensure consistency in ranking so that higher scores uniformly indicate better performance. We then compute z-scores for each metric, which standardizes the data based on the mean and standard deviation. This allows for comparison across different scales and distributions. We then make a list of existing updated metrics and use those to dynamically calculate the weights for each metric. We log the final weights and other relevant information in the ranking_config for future reference. We calculate a hybrid weighted penalty for missing metrics to account for incomplete or excluded data in the evaluation process. Our hybrid approach partially penalizes ignored metrics are excluded for valid reasons (like low variance), since their absence reduces the completeness of the evaluation, but fully penalizes missing metrics since those indicate NaNs when the wavelet should have values. We then calculate the weighted scores for both normalized and z-scored metrics. For the normalized scores, we do use the missing_metrics_count and penalty_weight to adjust the final score. We do not use this penalty though for calculating the z-score weighted score because the z-score already accounts for missing values and we want to use it to assess stability of scores. 
	
	Finally, we calculate the normalized difference between the norm-weighted score and z-score-weighted score for each wavelet. A large normalized difference suggests that the scores derived from two weighting methods (norm and z-score) diverge significantly, which might indicate instability in the scoring process, where a configuration’s performance is sensitive to how weights are applied. Conversely, a small normalized difference indicates that the configuration’s score remains consistent across different weighting methods, which suggests robustness in the configuration. By comparing the two methods, we can check for systematic biases in the configuration. After this calculation and sanity check, we compute a final stability-adjusted score. To calculate this, we use the norm-weighted score and then incorporate the normalized difference to penalize configurations with large inconsistencies. This ensures that configurations with smaller normalized differences are rewarded, as their scores are more reliable and less influenced by specific scoring methods and also prevents overfitting to a particular metric set. By applying a penalty proportional to the normalized difference, the stability-adjusted score balances high performance with consistency and helps us identify top-ranked configurations that excel across multiple perspectives. We use this score to rank the configurations and select the top N% of configurations. To ensure that we are getting not just the top results but also results for every wavelet, we also group them by wavelet, signal type, and wavelet type, and combine those top results with the top N% of configurations. Finally, we return the best configuration, ranked results, the top subset of ranked results, the overall correlation between normalized and z-score weighted scores, and the updated ranking configuration for future reference.
	
	Parameters:
	-----------
	results_df : pd.DataFrame
		DataFrame containing wavelet metrics.
	signal_type : str
		Type of signal being analyzed (e.g., "DWT", "CWT", "SWT").
	weights : dict, optional
		Dictionary of weights for each metric. Default is pre-defined.
	is_combined : bool, optional
		Flag indicating whether the results are combined (affects column prefix).
	epsilon_threshold : float, optional
		Threshold for log-transforming `wavelet_energy_entropy` due to negative values. Default is 1e-6, which is a small value to avoid division by zero.
	penalty_weight : float, optional
		Weight for penalizing missing metrics in the final score. Default is 0.05, which is a small penalty to avoid excessive influence on the final score. It is currently used for both norm-weighted scores and stability-adjusted scores.
	percentage_of_results : float, optional
		Percentage of top results to select. Default is 0.1, which selects the top 10% of configurations.

	Returns:
	--------
	best_config : pd.DataFrame
		The row containing the best wavelet configuration.
	ranked_results : pd.DataFrame
		DataFrame with combined scores and rankings.
	subset_ranked_results : pd.DataFrame
		DataFrame with combined scores and rankings for top configurations by wavelet, signal type, and wavelet type.
	overall_correlation_norm_zscore : float
		The correlation between normalized and z-score weighted scores.
	updated_ranking_config : dict
		Dictionary containing the ranking configuration for the analysis.
	"""
	processed_results = []
	reconstruction_ranking_configs = []
	for signal_type in original_signal_metrics_df["signal_type"].unique():
		console.print(f"Analyzing reconstruction accuracy for {signal_type} signal...")
		subset_results_df = results_df[results_df.signal_type == signal_type]
		subset_original_signal_metrics_df = original_signal_metrics_df[original_signal_metrics_df.signal_type == signal_type]
		ranked_reconstruction_df, reconstruction_config = compare_original_reconstructed_metrics(
			subset_original_signal_metrics_df,
			subset_results_df
		)
		reconstruction_ranking_configs.append(reconstruction_config)
		ranked_reconstruction_df.to_csv(f"ranked_reconstruction_df_{signal_type}.csv", index=False)
		# Merge reconstruction_score into results_df
		subset_results_df = subset_results_df.merge(
			ranked_reconstruction_df[["row_index", "reconstruction_score_weighted", "reconstruction_score_sum"]],
			left_index=True,
			right_on="row_index",
			how="left"
		)
		processed_results.append(subset_results_df)
	
	results_df = pd.concat(processed_results, ignore_index=True)
	results_df = results_df.drop(columns=["row_index"], errors="ignore")
	results_df = results_df.reset_index(drop=True)
	results_df.to_csv("results_df.csv", index=False)

	# Default weights if not provided
	if weights is None:
		weights = {
			# Core reconstruction metrics
			'wavelet_mse': 0.25,          # High importance for reconstruction accuracy
			'wavelet_psnr': 0.25,         # High importance for signal quality
			'emd_value': 0.2,             # High importance for reconstruction robustness
			'kl_divergence': 0.2,         # High importance for signal similarity
			# 'reconstruction_score': 0.4,   # Very high importance for overall performance

			# Wavelet-specific features
			'wavelet_energy_entropy': 0.15,  # Moderate importance for signal compression
			'wavelet_sparsity': 0.15,        # Moderate importance for sparsity
			'wavelet_entropy': 0.1,          # Moderate importance for signal decomposition efficiency

			# Signal-specific and derived metrics
			'smoothness': 0.05,              # Lower importance, aesthetic quality
			'correlation': 0.05,             # Lower importance, secondary robustness
			'avg_variance_across_levels': 0.1  # Balanced importance for decomposition consistency
		}
	ranking_config = {
		"signal_type": signal_type,
		"is_combined": is_combined,
		"metrics": [],
		"reconstruction_configs": reconstruction_ranking_configs
	}
	# Populate ranking_config dynamically
	for metric, original_weight in weights.items():
		ranking_config["metrics"].append({
			"metric": metric,
			"original_weight": original_weight,
			"final_weight": None,  # Will be updated later
			"normalized_weight": None,  # Will be updated later
			"ignore_metric": False,
			"removal_reason": None,
			"was_inverted": False,
			"was_shared": False,
			"was_specific": False,
			"variance": None,
			"presence": None,
			"was_zscored": False,
		})

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
				weights[metric] *= 0.5  # Reduce influence of low-variance metrics
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

	# Normalize metrics with RobustScaler for stability. Also filter metrics to exclude those flagged as ignored in the ranking_config
	metrics = [
		metric_config["metric"]
		for metric_config in ranking_config["metrics"]
		if not metric_config.get("ignore_metric", False)  # Exclude ignored metrics
		and metric_config["metric"] in results_df.columns  # Ensure existence in `results_df`
	]
	normalized_df = results_df.copy()
	scaler = RobustScaler()
	prefix = 'combined_' if is_combined else ''

	for metric in metrics:
		try:
			normalized_df[f"{prefix}{metric}_norm"] = scaler.fit_transform(results_df[[metric]])
		except ValueError as e:
			console.print(f"[bright_red]Error normalizing '{metric}': {e}. Skipping this metric.[/bright_red]")
			for metric_config in ranking_config["metrics"]:
				if metric_config["metric"] == metric:
					metric_config["ignore_metric"] = True
					metric_config["removal_reason"] = "Normalization Error"

	# Invert metrics where lower is better
	invert_metrics = ['wavelet_mse', 'wavelet_entropy', 'emd_value', 'kl_divergence']
	for metric in invert_metrics:
		if metric in metrics and f"{prefix}{metric}_norm" in normalized_df.columns:
			normalized_df[f"{prefix}{metric}_norm"] = 1 - normalized_df[f"{prefix}{metric}_norm"]
			for metric_config in ranking_config["metrics"]:
				if metric_config["metric"] == metric:
					metric_config["was_inverted"] = True

	# Compute z-scores
	for metric in metrics:
		if any(
			metric_config["metric"] == metric and metric_config["ignore_metric"]
			for metric_config in ranking_config["metrics"]
		):
			continue  # Skip ignored metrics
		# Proceed with z-score calculation
		metric_norm_col = f"{prefix}{metric}_norm"
		metric_zscore_col = f"{prefix}{metric}_zscore"
		
		std_dev = normalized_df[metric_norm_col].std()  # Calculate standard deviation
		if std_dev == 0 or pd.isna(std_dev):  # Handle cases with zero or NaN standard deviation
			console.print(f"[yellow]Standard deviation for {metric_norm_col} is zero or NaN. Skipping z-score calculation.[/yellow]")
			normalized_df[metric_zscore_col] = np.nan
		else:
			normalized_df[metric_zscore_col] = (
				normalized_df[metric_norm_col] - normalized_df[metric_norm_col].mean()
			) / std_dev
			for metric_config in ranking_config["metrics"]:
				if metric_config["metric"] == metric:
					metric_config["was_zscored"] = True

	# Normalize weights
	updated_metrics = [
		metric_config["metric"] for metric_config in ranking_config["metrics"] if not metric_config.get("ignore_metric", False) and metric_config["metric"] in results_df.columns
	]
	normalized_weights, updated_ranking_config = normalize_weights_dynamically(updated_metrics, weights, normalized_df, ranking_config)

	# Calculate weighted penalty for missing and ignored metrics
	normalized_df[f'{prefix}missing_metrics_count'] = normalized_df.apply(
		lambda row: sum(
			normalized_weights.get(metric, 0) * (1 if pd.isna(row[f"{prefix}{metric}_norm"]) else 0.5)
			for metric in updated_metrics
			if f"{prefix}{metric}_norm" in normalized_df.columns
			and (pd.isna(row[f"{prefix}{metric}_norm"]) or ranking_config["metrics"][updated_metrics.index(metric)]["ignore_metric"])
		),
		axis=1
	)
	
	# Compute weighted scores for norm and z-score
	ranking_config["penalty_weight"] = penalty_weight
	normalized_df[f"{prefix}wavelet_norm_weighted_score"] = normalized_df.apply(
		lambda row: (
			sum(
				normalized_weights[metric] * row[f"{prefix}{metric}_norm"]
				for metric in updated_metrics
				if pd.notna(row[f"{prefix}{metric}_norm"])
			) / max(sum(normalized_weights[metric] for metric in updated_metrics if pd.notna(row[f"{prefix}{metric}_norm"])), epsilon_threshold)
			- penalty_weight * row[f"{prefix}missing_metrics_count"]  # Use precomputed penalty
		),
		axis=1
	)

	normalized_df[f"{prefix}wavelet_zscore_weighted_score"] = normalized_df.apply(
		lambda row: (
			sum(
				normalized_weights[metric] * row[f"{prefix}{metric}_zscore"]
				for metric in updated_metrics
				if pd.notna(row[f"{prefix}{metric}_zscore"])
			) / max(sum(normalized_weights[metric] for metric in updated_metrics if pd.notna(row[f"{prefix}{metric}_zscore"])), epsilon_threshold)
		),
		axis=1
	)

	# Compute summed normalized score
	normalized_df[f"{prefix}wavelet_summed_norm_score"] = normalized_df.apply(
		lambda row: sum(
			row[f"{prefix}{metric}_norm"]
			for metric in updated_metrics
			if pd.notna(row[f"{prefix}{metric}_norm"])
		),
		axis=1
	)

	# Normalize summed score so that it has a comparable range
	max_summed_score = normalized_df[f"{prefix}wavelet_summed_norm_score"].max()
	if max_summed_score > 0:
		normalized_df[f"{prefix}wavelet_summed_norm_score"] /= max_summed_score
	else:
		console.print("[yellow]Max summed normalized score is zero! Assigning equal scores.[/yellow]")
		normalized_df[f"{prefix}wavelet_summed_norm_score"] = 1 / len(updated_metrics)  # Assign equal importance


	# Calculate normalized diff
	normalized_df[f"{prefix}normalized_diff"] = (
		(normalized_df[f"{prefix}wavelet_norm_weighted_score"] - normalized_df[f"{prefix}wavelet_zscore_weighted_score"]).abs()
		/ (normalized_df[f"{prefix}wavelet_norm_weighted_score"] + normalized_df[f"{prefix}wavelet_zscore_weighted_score"]).abs()
	)

	# Final stability-adjusted score
	normalized_df[f"{prefix}final_score"] = (
		normalized_df[f"{prefix}wavelet_norm_weighted_score"]
		- penalty_weight * normalized_df[f"{prefix}normalized_diff"]
	)

	# Rank results by final_score (Weighted method)
	ranked_results = normalized_df.sort_values(
		by=f"{prefix}final_score", ascending=False
	).reset_index(drop=True)
	ranked_results[f"{prefix}wavelet_rank"] = ranked_results.index + 1

	# Rank results by summed normalized score (Summing method)
	ranked_results = ranked_results.sort_values(
		by=f"{prefix}wavelet_summed_norm_score", ascending=False
	).reset_index(drop=True)
	ranked_results[f"{prefix}wavelet_summed_rank"] = ranked_results.index + 1

	ranked_results.to_csv(f"test_{signal_type}_ranked_results.csv", index=False)

	# Dynamically select the top N% of ranked results
	num_top_results = max(1, int(len(ranked_results) * percentage_of_results))  # At least one result
	top_ranked_results = ranked_results.head(num_top_results)

	# Select top configurations by wavelet, signal type, and wavelet type
	grouping_cols = ['wavelet_type', 'wavelet'] if is_combined else ['wavelet']
	grouped = ranked_results.groupby(grouping_cols)

	subset_ranked_results = grouped.apply(
		lambda group: group.loc[group[f"{prefix}final_score"].idxmax()]
	).reset_index(drop=True)

	drop_cols = ['signal_type', 'wavelet_type', 'wavelet', 'wavelet_mode', 'wavelet_level']
	drop_cols = [drop_col for drop_col in drop_cols if (drop_col in subset_ranked_results.columns) and (drop_col in top_ranked_results.columns)]
	final_ranked_results = pd.concat([top_ranked_results, subset_ranked_results], ignore_index=True).drop_duplicates(subset=drop_cols, keep='first')
	final_ranked_results = final_ranked_results.sort_values(
		by=f"{prefix}final_score", ascending=False
	).reset_index(drop=True)
	final_ranked_results[f"{prefix}final_wavelet_rank"] = final_ranked_results.index + 1
	if final_ranked_results[
		[f"{prefix}wavelet_norm_weighted_score", f"{prefix}wavelet_zscore_weighted_score"]
	].dropna().shape[0] == 0:
		overall_correlation_norm_zscore = float("nan")
		console.print("[yellow]Warning: Insufficient data for correlation calculation. Setting to NaN.[/yellow]")
	else:
		overall_correlation_norm_zscore = final_ranked_results[
			[f"{prefix}wavelet_norm_weighted_score", f"{prefix}wavelet_zscore_weighted_score"]
		].corr().iloc[0, 1]
	# Return the best configuration, rankings, and correlation
	best_config = final_ranked_results.iloc[0:1]
	return best_config, ranked_results, final_ranked_results, overall_correlation_norm_zscore, updated_ranking_config

def calculate_rank_stability(df: pd.DataFrame, rank_columns: list) -> pd.DataFrame:
	"""
	Calculate a stability metric for wavelet rankings based on multiple ranking columns. The rank columns list should be ordered from the least to most important rank.
	
	Parameters:
	-----------
	df : pd.DataFrame
		DataFrame containing rank columns to evaluate.
	rank_columns : list of str
		Columns representing ranks to compare for stability.
		
	Returns:
	--------
	pd.DataFrame
		DataFrame with an added 'rank_stability' column.
	"""
	# Compute absolute differences between ranks
	for i, col_a in enumerate(rank_columns):
		for col_b in rank_columns[i+1:]:
			diff_col_name = f"{col_a}_vs_{col_b}_abs_diff"
			df[diff_col_name] = (df[col_a] - df[col_b]).abs()
	
	# Calculate the standard deviation of ranks across rank columns
	df['rank_std_dev'] = df[rank_columns].std(axis=1)
	
	# Normalize by the maximum possible rank
	max_rank = df[rank_columns].max().max()
	df['rank_stability'] = 1 - (df['rank_std_dev'] / max_rank)
	
	return df

def compute_wavelet_scores(df: pd.DataFrame, is_combined: bool, rank_bins:list=[0, 10, 20, 50, 100, None],):
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

	prefix = 'all_volumes_' if is_combined else 'individual_volume_'
	
	# Normalize rank and stability
	df[f'{prefix}normalized_rank'] = df['combined_final_wavelet_rank'] / df['combined_final_wavelet_rank'].max()
	df[f'{prefix}normalized_stability'] = 1 - df[f'rank_stability']

	# Define weights for rank and stability
	alpha = 0.5  # Weight for rank
	beta = 0.5   # Weight for stability

	# Compute composite score
	df[f'{prefix}composite_score'] = (
		alpha * df[f'{prefix}normalized_rank'] + beta * df[f'{prefix}normalized_stability']
	)
	
	# Assign rank bins
	if rank_bins[-1] is None:  # Replace None with max rank
		rank_bins[-1] = df['combined_final_wavelet_rank'].max()

	df[f'{prefix}rank_bin'] = pd.cut(
		df['combined_final_wavelet_rank'],
		bins=rank_bins,
		labels=[f"Top {int(bin_edge)}" if bin_edge != rank_bins[-1] else "Beyond 100" for bin_edge in rank_bins[1:]]
	)
	# Add rank bin summaries
	rank_bin_summary = df.groupby(['wavelet_family', f'{prefix}rank_bin']).agg(
		count=('combined_final_wavelet_rank', 'count'),
		unique_htid=('htid', 'nunique'),  # Count of unique volumes (htid)
		binned_mean_rank_stability=(f'rank_stability', 'mean'),  # Mean rank stability
		binned_std_rank_stability=(f'rank_stability', 'std')  # Standard deviation of rank stability
	).reset_index()

	# Add proportions
	rank_bin_summary[f'{prefix}global_proportion'] = rank_bin_summary['count'] / rank_bin_summary.groupby(f'{prefix}rank_bin')['count'].transform('sum')
	rank_bin_summary[f'{prefix}htid_proportion'] = rank_bin_summary['unique_htid'] / rank_bin_summary.groupby(f'{prefix}rank_bin')['unique_htid'].transform('sum')

	top10_metrics = rank_bin_summary[rank_bin_summary[f'{prefix}rank_bin'] == 'Top 10'].sort_values(by=[f'{prefix}global_proportion', f'{prefix}htid_proportion', 'mean_rank_stability', 'std_rank_stability', 'count', 'unique_htid'], ascending=[False, False, False, True, False, False])

	# Aggregate metrics based on the level
	wavelet_summary = df.groupby('wavelet_family').agg(
		mean_composite_score=(f'{prefix}composite_score', 'mean'),
		mean_rank_stability=(f'rank_stability', 'mean'),
		std_rank_stability=(f'rank_stability', 'std'),
		mean_rank=('combined_final_wavelet_rank', 'mean'),
		total_count=('htid', 'count')
	).reset_index()

	wavelet_summary = wavelet_summary.merge(top10_metrics, on='wavelet_family', how='left').fillna(0)	

	# Normalize all metrics
	for col in ['mean_composite_score', 'mean_rank_stability', 'mean_rank', 'total_count', f'{prefix}global_proportion', f'{prefix}htid_proportion']:
		wavelet_summary[f'{prefix}normalized_{col}'] = wavelet_summary[col] / wavelet_summary[col].max()

	# Compute final composite score
	wavelet_summary[f'{prefix}wavelet_composite_score'] = (
		0.3 * wavelet_summary[f'{prefix}normalized_mean_composite_score'] +
		0.25 * wavelet_summary[f'{prefix}normalized_mean_rank_stability'] +
		0.15 * wavelet_summary[f'{prefix}normalized_mean_rank'] +
		0.1 * wavelet_summary[f'{prefix}normalized_total_count'] +
		0.1 * wavelet_summary[f'{prefix}normalized_global_proportion'] +
		0.1 * wavelet_summary[f'{prefix}normalized_htid_proportion']
	)

	# Sort by final score
	wavelet_summary = wavelet_summary.sort_values(by=f'{prefix}_wavelet_composite_score', ascending=False)

	# Select the best wavelet
	top_wavelet_family = wavelet_summary.iloc[0]
	console.print(f"Best wavelet family: {top_wavelet_family.wavelet_family}", style="bright_magenta")

	final_df = df.merge(rank_bin_summary, on=['wavelet_family', f'{prefix}rank_bin'], how='left')
	final_df[f'{prefix}top_wavelet_family'] = top_wavelet_family
	final_df = final_df.sort_values(by=[f'{prefix}composite_score'], ascending=[False])
	return final_df

def generate_finalized_wavelets():
	data_directory_path = get_data_directory_path()
	preidentified_periodicals_df = read_csv_file(os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", "periodical_metadata", "classified_preidentified_periodicals_with_full_metadata.csv"))

	all_frequencies_df = pd.read_csv("../datasets/all_volume_features_and_frequencies.csv")
	console.print(f"Processed {len(all_frequencies_df)} volume features and frequencies.", style="bright_green")

	missing_volumes = preidentified_periodicals_df[~preidentified_periodicals_df.htid.isin(all_frequencies_df.htid)]
	console.print(f"Missing Volumes: {len(missing_volumes)}", style="bright_red")
	missing_titles = missing_volumes.lowercase_periodical_name.unique().tolist()
	console.print(f"Missing Periodical Titles: {missing_titles}", style="bright_red")
	  
	existing_titles = preidentified_periodicals_df.lowercase_periodical_name.unique().tolist()
	rank_columns = ['wavelet_rank', 'final_wavelet_rank', 'combined_wavelet_rank', 'combined_final_wavelet_rank']
	for index, periodical_title in enumerate(existing_titles):
		console.print(f"Processing {periodical_title} ({index + 1}/{len(existing_titles)})...", style="bright_yellow")
		subset_preidentified_periodicals_df = preidentified_periodicals_df[(preidentified_periodicals_df['lowercase_periodical_name'] == periodical_title) & (preidentified_periodicals_df.volume_directory.notna())]
		console.print(f"Processed {len(subset_preidentified_periodicals_df)} volumes for {periodical_title}.", style="bright_green")
		if len(subset_preidentified_periodicals_df) == 0:
			continue
		

		volume_dfs = []
		for index, row in subset_preidentified_periodicals_df.iterrows():
			individual_htid = row.htid
			individual_publication_directory = row.publication_directory
			individual_volume_directory = row.volume_directory
			console.print(f"Individual HTID: {individual_htid}", style="bright_green")
			console.print(f"Individual Publication Directory: {individual_publication_directory}", style="bright_green")
			console.print(f"Individual Volume Directory: {individual_volume_directory}", style="bright_green")
			subset_frequencies_df = all_frequencies_df[all_frequencies_df.htid == individual_htid]
			console.print(f"Processed {len(subset_frequencies_df)} frequencies for {individual_htid}.", style="bright_green")

			subset_combined_results_path = os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", individual_publication_directory, "volumes", individual_volume_directory, "wavelet_analysis", individual_volume_directory + "_subset_combined_results.csv")
			if os.path.exists(subset_combined_results_path):
				subset_combined_results_df = pd.read_csv(subset_combined_results_path)
				subset_combined_results_df['htid'] = individual_htid
				console.print(f"Loaded {len(subset_combined_results_df)} subset combined results from {subset_combined_results_path}.", style="bright_green")
			else:
				console.print(f"Could not find {subset_combined_results_path}.", style="bright_red")

				wavelet_volume_data_path = os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", individual_publication_directory, "volumes", individual_volume_directory, "wavelet_analysis", individual_volume_directory + "_wavelet_volume_results.csv")
				if os.path.exists(wavelet_volume_data_path):
					wavelet_volume_data_df = pd.read_csv(wavelet_volume_data_path)
					console.print(f"Loaded {len(wavelet_volume_data_df)} wavelet volume data from {wavelet_volume_data_path}.", style="bright_green")
				else:
					console.print(f"Could not find {wavelet_volume_data_path}.", style="bright_red")
	
			if not wavelet_volume_data_df.empty and not subset_combined_results_df.empty:
				
				shared_cols = set(subset_combined_results_df.columns).intersection(set(wavelet_volume_data_df.columns))
				avoid_cols = [col for col in wavelet_volume_data_df.columns if not col in shared_cols]
				final_cols = avoid_cols + ['htid']
				subset_combined_results_df = subset_combined_results_df.merge(wavelet_volume_data_df[final_cols], on='htid', how='left')
				subset_combined_results_df['wavelet_family'] = subset_combined_results_df['wavelet'].str.extract(r'([a-zA-Z]+)')

				subset_combined_results_df = calculate_rank_stability(subset_combined_results_df, rank_columns)
	
				finalized_subset_combined_results_df = compute_wavelet_scores(subset_combined_results_df, is_combined=False)
				if finalized_subset_combined_results_df is not None:
					# Save the finalized subset combined results
					finalized_subset_outputh_path = os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", individual_publication_directory, "volumes", individual_volume_directory, "wavelet_analysis", individual_volume_directory + "_finalized_subset_combined_results.csv")	
					finalized_subset_combined_results_df.to_csv(finalized_subset_outputh_path, index=False)
					volume_dfs.append(finalized_subset_combined_results_df)
			# Combine all volume data for the title into one DataFrame
		# Combine all volumes for the title
		combined_volume_df = pd.concat(volume_dfs, ignore_index=True)

		# Compute wavelet scores for the combined title
		final_combined_volume_df = compute_wavelet_scores(combined_volume_df, is_combined=True)
		if final_combined_volume_df is not None:
			# Save the finalized combined results
			finalized_combined_output_path = os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", individual_publication_directory, "derived_data", "wavelet_analysis", periodical_title + "_finalized_combined_results.csv")
			os.makedirs(os.path.dirname(finalized_combined_output_path), exist_ok=True)	
			final_combined_volume_df.to_csv(finalized_combined_output_path, index=False)
			console.print(f"Saved finalized combined results for {periodical_title} to {finalized_combined_output_path}.", style="bright_green")


if __name__ == "__main__":
	generate_finalized_wavelets()