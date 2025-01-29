# Standard library imports
import os
import sys
import warnings
import shutil
import json

# Third-party imports
import pandas as pd
import numpy as np
import altair as alt
from tqdm import tqdm
from rich.console import Console
import pywt
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import detrend
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from difflib import SequenceMatcher
from fastdtw import fastdtw
from minineedle import needle, core

# Local application imports
sys.path.append("../..")
from scripts.utils import read_csv_file, get_data_directory_path, save_chart, generate_table, process_tokens
from scripts.wavelet_scripts.generate_wavelet_stationarity import preprocess_signal_for_stationarity, check_wavelet_stationarity
from scripts.wavelet_scripts.generate_wavelet_signal_processing import evaluate_dwt_performance, evaluate_dwt_performance_parallel, evaluate_cwt_performance, evaluate_cwt_performance_parallel, evaluate_swt_performance, evaluate_swt_performance_parallel, calculate_signal_metrics
from scripts.wavelet_scripts.generate_wavelet_plots import plot_volume_frequencies_matplotlib, plot_tokens_per_page, plot_annotated_periodicals

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

## WAVELET RANKING AND COMPARISON FUNCTIONS
def filter_wavelets(wavelets: list, exclude_complex: bool = True) -> list:
	"""
	Filter out complex-valued wavelets if needed.

	Parameters:
	-----------
	wavelets : list
		List of wavelet names to test.
	exclude_complex : bool
		Whether to exclude complex-valued wavelets.

	Returns:
	--------
	filtered_wavelets : list
		List of filtered wavelets.
	"""
	complex_wavelets = ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7']
	if exclude_complex:
		filtered_wavelets = [w for w in wavelets if w not in complex_wavelets]
		console.print(f"[yellow]Excluding complex-valued wavelets: {complex_wavelets}[/yellow]")
		return filtered_wavelets
	return wavelets

def normalize_weights_dynamically(existing_metrics: list, weights: dict, results_df: pd.DataFrame, ranking_config: dict, threshold:float=0.9, shared_weight_factor:float=0.7, specific_weight_factor:float=0.3) -> tuple:
	"""
	This function dynamically normalize weights for metrics based on variance, presence, and distribution across shared and specific metrics. It ensures that metrics are appropriately prioritized, reflecting their relevance and availability, while also maintaining consistency across all metrics. It first checks the variance and presence of each metric. Variance reflects the amount of variation in a metric across the dataset. Metrics with low variance may not provide meaningful distinctions between wavelet configurations, as they exhibit minimal variability. Presence indicates the proportion of data rows where the metric is non-null. Metrics with higher presence values are more widely applicable and thus hold more weight in decision-making. 
	
	The function then identifies shared and specific metrics based on a predefined threshold. Shared metrics are those with a presence above the threshold, while specific metrics have a presence below the threshold. Shared metrics are prioritized over specific metrics, as they are more widely applicable and thus more likely to influence the final ranking. Weights are dynamically adjusted by combining the initial weight with a factor derived from the metric’s variance and presence. Shared metrics receive a higher proportion (e.g., 70%) of the total weight allocation, reflecting their broader relevance, while specific metrics are allocated the remaining proportion (e.g., 30%). The final weights are normalized to sum to 1, ensuring that the total weight is consistent across all metrics. This is crucial for proportional calculations when combining metric scores. Without normalization, metrics with higher raw weights could disproportionately skew results. As a final sanity check, the code loops through the weights and checks if any have been weighted to zero. If that happens it raises an error, as it indicates an unexpected omission or a potential issue with the weighting logic and stops the process. As long as there are no zeros, the function logs the final weights and other relevant information in the ranking configuration for future reference, and then returns the normalized weights and updated ranking configuration.

	Parameters:
	-----------
	metrics : list
		List of metric names to consider.
	weights : dict
		Dictionary of initial weights for each metric.
	results_df : pd.DataFrame
		DataFrame containing the results with metrics.
	ranking_config : dict
		Configuration dictionary to log weight adjustments.
	threshold : float, optional
		Threshold for presence to distinguish shared and specific metrics. Default is 0.9, which means metrics with a presence of 90% or higher are considered shared.
	shared_weight_factor : float, optional
		Factor to prioritize shared metrics. Default is 0.7, which means shared metrics receive 70% of the total weight allocation.
	specific_weight_factor : float, optional
		Factor to prioritize specific metrics. Default is 0.3, which means specific metrics receive 30% of the total weight allocation.

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

	# Log variance and presence for each metric in ranking_config
	for metric in metrics:
		for metric_config in ranking_config["metrics"]:
			if metric_config["metric"] == metric:
				metric_config["variance"] = metric_variances.get(metric, None)
				metric_config["presence"] = metric_presence.get(metric, None)

	# Step 2: Identify Shared and Specific Metrics
	shared_metrics = [
		metric for metric in metrics if metric_presence.get(metric, 0) >= threshold
	]
	specific_metrics = [metric for metric in metrics if metric not in shared_metrics]

	# If all metrics are removed or ignored, fallback to even distribution or raise an error.
	if not shared_metrics and not specific_metrics:
		raise ValueError("No valid metrics to normalize. Please check metric definitions.")

	# Step 3: Adjust Weights Dynamically
	dynamic_adjustments = {
		metric: metric_variances[metric] * metric_presence[metric]
		for metric in metrics
		if metric in weights
	}

	# Step 4: Normalize Weights
	normalized_weights = {}
	for metric in shared_metrics:
		normalized_weights[metric] = (
			weights[metric] * dynamic_adjustments.get(metric, 1.0)
		)
	for metric in specific_metrics:
		normalized_weights[metric] = (
			weights[metric] * dynamic_adjustments.get(metric, 1.0)
		)

	# Normalize weights within shared and specific buckets
	shared_total_weight = sum(normalized_weights[metric] for metric in shared_metrics)
	specific_total_weight = sum(normalized_weights[metric] for metric in specific_metrics)

	for metric in shared_metrics:
		normalized_weights[metric] = (
			normalized_weights[metric] / shared_total_weight * shared_weight_factor
		)
		for metric_config in ranking_config["metrics"]:
			if metric_config["metric"] == metric:
				metric_config["was_shared"] = True

	for metric in specific_metrics:
		normalized_weights[metric] = (
			normalized_weights[metric] / specific_total_weight * specific_weight_factor
		)
		for metric_config in ranking_config["metrics"]:
			if metric_config["metric"] == metric:
				metric_config["was_specific"] = True

	# Normalize all weights to sum to 1
	total_weight = sum(normalized_weights.values())
	if total_weight > 0:
		normalized_weights = {metric: weight / total_weight for metric, weight in normalized_weights.items()}
	else:
		console.print("[bright_red]All weights removed! Check your metric definitions.[/bright_red]")
		normalized_weights = {metric: 1 / len(metrics) for metric in metrics}  # Fallback: Equal distribution
		console.print("[yellow]Fallback: Weights evenly distributed across metrics.[/yellow]")

	# Step 5: Validate Weights and Stop on Critical Issues
	for metric in metrics:
		weight = normalized_weights.get(metric, 0)
		if weight == 0:
			raise ValueError(
				f"Critical Error: Metric '{metric}' has a weight of zero in normalized weights. "
				"This indicates an unexpected omission or a potential issue with the weighting logic."
			)

	# Step 6: Update ranking_config with final weights
	for metric, final_weight in normalized_weights.items():
		for metric_config in ranking_config["metrics"]:
			if metric_config["metric"] == metric:
				metric_config["final_weight"] = final_weight


	return normalized_weights, ranking_config

def run_global_sequence_alignment(target_sequence: list, window: list, placeholder: int = -1) -> int:
	"""
	Applies global sequence alignment on the implied zero values within a window using minineedle, with placeholders.
	
	Parameters
	----------
	target_sequence : list
		The target sequence to be used in the alignment.
	window : list
		The window containing the observed sequence.
	placeholder : int
		The placeholder value to be used in the alignment.
		
	Returns
	-------
	alignment_score : int
		The alignment score. If the alignment fails, returns 0. A higher score indicates a better alignment.
	"""
	# Extract the observed sequence from the window
	observed_sequence = [float(p) if pd.notna(p) else placeholder for p in window]
	
	# Check for valid entries in the observed sequence
	if all(val == placeholder for val in observed_sequence):
		return 0

	# Create Needleman-Wunsch global alignment instance
	alignment = needle.NeedlemanWunsch(observed_sequence, target_sequence)
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
            print(f"DTW distance: {distance}")
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
def process_array(original_value, reconstructed_value):
	"""
	Convert stringified or list-like arrays to numpy arrays.
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
	original_df: pd.DataFrame, reconstructed_df: pd.DataFrame, cosine_weight: float = 0.7, diff_weight: float = 0.3
) -> pd.DataFrame:
	"""
	Compare original and reconstructed signal metrics row-by-row, compute total scores, and rank results.
	"""
	if original_df.empty or reconstructed_df.empty:
		console.print("[yellow]One or both DataFrames are empty. Returning empty results.[/yellow]")
		return pd.DataFrame()

	if len(original_df) != 1:
		raise ValueError("original_df must contain exactly one row for comparison.")

	comparison_rows = []  # To store row-by-row comparison results
	metrics = set(original_df.columns).intersection(set(reconstructed_df.columns)) - {"signal_type"}

	for idx, reconstructed_row in reconstructed_df.iterrows():
		comparison = {"row_index": idx}  # Track the row index for reference

		for metric in metrics:
			if metric in original_df.columns and metric in reconstructed_df.columns:
				console.print(f"Comparing metric: {metric}")

				if isinstance(original_df[metric].iloc[0], (list, np.ndarray)) or (
					isinstance(original_df[metric].iloc[0], str) and original_df[metric].iloc[0].startswith("[")
				):
					original_array, reconstructed_array = process_array(original_df[metric].iloc[0], reconstructed_row[metric])

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
	scaler = MinMaxScaler()
	lower_is_better_metrics = [
		col for col in comparison_df.columns if "diff" in col or col in [
			"wasserstein", "dtw", "euclidean", "avg_diff", "total_diff"
		]
	]
	for col in comparison_df.columns:
		if col.endswith("_normalized") or col == "row_index":
			continue  # Skip already normalized columns or non-metric columns

		# Normalize metrics where lower values are better
		if col in lower_is_better_metrics:
			# Invert the relationship (higher is better)
			max_value = comparison_df[col].max()
			normalized_df[col + "_normalized"] = 1 - (comparison_df[col] / max_value)
		else:
			# Direct normalization for metrics where higher is better
			normalized_df[col + "_normalized"] = scaler.fit_transform(comparison_df[col].values.reshape(-1, 1)).flatten()

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
	total_scores = []
	for _, row in normalized_df.iterrows():
		weighted_scores = []
		for metric_type, weight in metric_weights.items():
			metric_columns = [col for col in normalized_df.columns if col.endswith(metric_type + "_normalized")]
			if metric_columns:
				metric_score = row[metric_columns].mean()  # Average across related normalized metrics
				weighted_scores.append(weight * metric_score)

		total_scores.append(sum(weighted_scores))

	normalized_df["reconstruction_score"] = total_scores
	# total_scores = []
	# for _, row in comparison_df.iterrows():
	# 	# Aggregate scores for each metric type
	# 	weighted_scores = []
	# 	for metric_type, weight in metric_weights.items():
	# 		# Find relevant columns for the metric type
	# 		metric_columns = [col for col in comparison_df.columns if col.endswith(metric_type)]
	# 		if metric_columns:
	# 			# Average score for the metric type across all relevant columns
	# 			metric_score = np.nanmean([row[col] for col in metric_columns])
	# 			# Normalize distance-based scores (e.g., DTW) to align with similarity-based scores
	# 			if metric_type in {"dtw", "euclidean", "wasserstein", "total_diff", "avg_diff"} and metric_score is not None:
	# 				metric_score = 1 / (1 + metric_score)  # Inverse of distance for scoring
	# 			if metric_score is not None:
	# 				weighted_scores.append(weight * metric_score)

	# 	# Sum up weighted scores to compute the total score
	# 	total_score = sum(weighted_scores)
	# 	total_scores.append(total_score)

	# # Add total scores to the DataFrame
	# comparison_df["reconstruction_score"] = total_scores
	# Correlation Analysis
	correlation_matrix = normalized_df.corr()
	reconstruction_corr = correlation_matrix["reconstruction_score"].sort_values(ascending=False)

	console.print(f"Reconstruction Score Correlation Analysis:\n{reconstruction_corr}")

	# Sort by total score in descending order
	ranked_comparison_df = normalized_df.sort_values(by="reconstruction_score", ascending=False).reset_index(drop=True)
	return ranked_comparison_df

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
	for signal_type in original_signal_metrics_df["signal_type"].unique():
		console.print(f"Analyzing reconstruction accuracy for {signal_type} signal...")
		results_df[results_df.signal_type == signal_type].to_csv(f"test_{signal_type}_results.csv", index=False)
		original_signal_metrics_df[original_signal_metrics_df.signal_type == signal_type].to_csv(f"test_{signal_type}_original_signal_metrics.csv", index=False)
		ranked_reconstruction_df = compare_original_reconstructed_metrics(
			original_signal_metrics_df[original_signal_metrics_df.signal_type == signal_type],
			results_df[results_df.signal_type == signal_type]
		)
		ranked_reconstruction_df.to_csv(f"test_results_{signal_type}.csv", index=False)

		# Merge reconstruction_score into results_df
		results_df = results_df.merge(
			ranked_reconstruction_df[["row_index", "reconstruction_score"]],
			left_index=True,
			right_on="row_index",
			how="left"
		)

	# Default weights if not provided
	if weights is None:
		weights = {
			# Core reconstruction metrics
			'wavelet_mse': 0.25,          # High importance for reconstruction accuracy
			'wavelet_psnr': 0.25,         # High importance for signal quality
			'emd_value': 0.2,             # High importance for reconstruction robustness
			'kl_divergence': 0.2,         # High importance for signal similarity

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
		"metrics": []
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

	# Rank results
	ranked_results = normalized_df.sort_values(
		by=f"{prefix}final_score", ascending=False
	).reset_index(drop=True)
	ranked_results[f"{prefix}wavelet_rank"] = ranked_results.index + 1

	# Dynamically select the top N% of ranked results
	num_top_results = max(1, int(len(ranked_results) * percentage_of_results))  # At least one result
	top_ranked_results = ranked_results.head(num_top_results)

	# Select top configurations by wavelet, signal type, and wavelet type
	grouping_cols = ['wavelet_type', 'wavelet'] if is_combined else ['wavelet']
	grouped = ranked_results.groupby(grouping_cols)

	subset_ranked_results = grouped.apply(
		lambda group: group.loc[group[f"{prefix}final_score"].idxmax()]
	).reset_index(drop=True)

	final_ranked_results = pd.concat([top_ranked_results, subset_ranked_results], ignore_index=True).drop_duplicates()
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

def compare_and_rank_wavelet_metrics(
	raw_signal: np.ndarray,
	smoothed_signal: np.ndarray,
	wavelet_directory: str,
	volume_id: str,
	signal_metrics_df: pd.DataFrame,
	wavelet_transform_settings: dict,
	use_parallel: bool = True,
	weights: dict = None,
	compare_top_subset: bool = True,
) -> pd.DataFrame:
	"""
	This function evaluates wavelet performance for a raw signal and its smoothed counterpart across three types of wavelet transforms: DWT (Discrete Wavelet Transform), CWT (Continuous Wavelet Transform), and SWT (Stationary Wavelet Transform). Its goal is to rank and combine configurations to determine the best overall wavelet representation for the signal.

	It first starts by defining ranking weights if not provided for metrics such as MSE, PSNR, energy entropy, and sparsity. It then retrieves a list of available DWT and CWT wavelets using PyWavelets library. CWT wavelets are filtered to exclude complex-valued wavelets due to current implementation constraints. It also creates a file path for saving results that is constructed using the specified wavelet_directory and volume_id.
	
	For each wavelet type (DWT, CWT, and SWT), the function processes both the raw and smoothed signals. Evaluations can run in parallel or sequentially, controlled by the use_parallel flag. Parallelization can speed up large-scale computations. If any errors occur during evaluation, the function logs the error and initializes the results as empty DataFrames. Results include both successfully evaluated configurations and skipped configurations (e.g., unsupported parameter combinations). Skipped configurations are saved for transparency once combined between the raw and smoothed signals. 

	For the successfully performed wavelets, results for raw and smoothed signals are combined and are passed to the determine_best_wavelet_representation function to rank configurations based on the provided metrics and weights. That function returns the best configuration, ranked results, the top subset of ranked results, the overall correlation between normalized and z-score weighted scores, and the updated ranking configuration for future reference. We save the full final results, the top subset of results, and the ranking configuration to separate files for each wavelet type and signal type. Finally, a table is generated with summarizing the best configuration for each wavelet type.

	The next step is to repeat the process for the results from DWT, CWT, and SWT, which are combined into a single DataFrame. If compare_top_subset is set to True, only the top subset of results is compared, which is the default primarily for speed and efficiency. The top subset is typically sufficient for identifying the best configurations, and it reduces the computational burden of processing the full results. The combined results are then ranked and compared to determine the best overall wavelet representation across all types, with again all three files saved. The best combined results are returned, which contain the best wavelet configuration.
	"""
	modes = pywt.Modes.modes
	# Define wavelet types and their evaluation functions
	wavelet_types = {
		"DWT": {
			"wavelets": pywt.wavelist(kind='discrete'),
			"evaluate": evaluate_dwt_performance_parallel if use_parallel else evaluate_dwt_performance,
		},
		"CWT": {
			"wavelets": filter_wavelets(pywt.wavelist(kind='continuous')),
			"evaluate": evaluate_cwt_performance_parallel if use_parallel else evaluate_cwt_performance,
		},
		"SWT": {
			"wavelets": pywt.wavelist(kind='discrete'),
			"evaluate": evaluate_swt_performance_parallel if use_parallel else evaluate_swt_performance,
		},
	}

	# Helper function to process wavelets
	def process_wavelet_type(wavelet_type, wavelet_info, signal, modes, signal_type):
		# try:
		print(f"Processing {wavelet_type} wavelets...")
		if wavelet_type == "DWT":
			results, skipped_results = wavelet_info["evaluate"](signal, wavelet_info["wavelets"], modes, signal_type)
			
		else:
			results, skipped_results = wavelet_info["evaluate"](signal, wavelet_info["wavelets"], signal_type)
		# except Exception as e:
		# 	console.print(f"[bright_red]Error evaluating {wavelet_type}: {e}[/bright_red]")
		# 	results, skipped_results = pd.DataFrame(), pd.DataFrame()
		return results, skipped_results

	# Collect results across wavelet types and signals
	all_results = []
	file_path = wavelet_directory + f"/{volume_id.replace('.', '_')}_"
	console.print(f"[bright_cyan]Processing file path {file_path}[/bright_cyan]")
	# Ensure results are non-empty before ranking
	table_cols = ['wavelet_rank', 'final_wavelet_rank', 'final_score', 'wavelet_norm_weighted_score', 'normalized_diff', 'wavelet_zscore_weighted_score', 'missing_metrics_count']
	for wavelet_type, wavelet_info in wavelet_types.items():
		console.print(f"[blue]Processing {wavelet_type} wavelet type[/blue]")
		wavelet_results = []
		wavelet_skipped_results = []
		for signal_type, settings in wavelet_transform_settings.items():
			signal = raw_signal if signal_type == 'raw' else smoothed_signal
			console.print(f"[blue]  Processing {signal_type} signal for {wavelet_type}[/blue]")

			# Skip DWT and SWT for non-stationary signals
			if wavelet_type in ["DWT", "SWT"] and not settings["is_stationary"]:
				console.print(f"[yellow]  Skipping {wavelet_type} for {signal_type} signal (non-stationary).[/yellow]")
				continue

			# Process the wavelet type
			results, skipped_results = process_wavelet_type(wavelet_type, wavelet_info, signal, modes, signal_type)
			wavelet_results.append(results)
			wavelet_skipped_results.append(skipped_results)
		# Save skipped results
		results_df = pd.concat(wavelet_results, ignore_index=True)
		skipped_df = pd.concat(wavelet_skipped_results, ignore_index=True)
		if not skipped_df.empty:
			skipped_df.to_csv(
				f"{file_path}{wavelet_type.lower()}_skipped_results.csv",
				index=False
			)
		print(f"Results {len(results_df)} for {wavelet_type} Wavelet Type")
		# Save results and rank them
		if not results_df.empty:
			results_df['wavelet_type'] = wavelet_type
			best_config, ranked, subset_ranked, correlation_score, ranking_config = determine_best_wavelet_representation(
				results_df, wavelet_type, signal_metrics_df, weights, False
			)
			
			suffix = f"{wavelet_type.lower()}_{signal_type}"
			ranked.to_csv(f"{file_path}full_{wavelet_type.lower()}_results.csv", index=False)
			subset_ranked.to_csv(f"{file_path}subset_{wavelet_type.lower()}_results.csv.csv", index=False)
			with open(f"{file_path}{wavelet_type.lower()}_ranking_config.json", "w") as f:
				json.dump(ranking_config, f, indent=4)
			generate_table(best_config[ ['wavelet', 'signal_type'] + table_cols], f"Best {wavelet_type} Wavelet Configuration (Correlation: {correlation_score:.2f})")
			# Append to all_results
			if compare_top_subset:
				subset_ranked['wavelet_type'] = wavelet_type
				all_results.append(subset_ranked)
			else:
				ranked['wavelet_type'] = wavelet_type
				all_results.append(ranked)

	# Combine all results
	combined_all_results = pd.concat(all_results, ignore_index=True)

	# Perform final ranking
	if not combined_all_results.empty:
		best_combined, ranked_combined, subset_combined, combined_correlation, combined_config = determine_best_wavelet_representation(
			combined_all_results, "Combined", signal_metrics_df, weights, True
		)
		suffix = "" if compare_top_subset else "all_"
		ranked_combined.to_csv(f"{file_path}combined_{suffix}results.csv", index=False)
		subset_combined.to_csv(f"{file_path}combined_{suffix}subset.csv", index=False)
		with open(f"{file_path}combined_{suffix}ranking_config.json", "w") as f:
			json.dump(combined_config, f, indent=4)
		generate_table(best_combined[ ['wavelet', 'signal_type'] + table_cols], f"Best Combined Wavelet Configuration (Correlation: {combined_correlation:.2f})")
		return best_combined
	else:
		console.print("[red]No valid wavelet configurations found.[/red]")
		return pd.DataFrame()

## MAIN FUNCTIONS

def generate_signal_processing_data(volume_paths_df: pd.DataFrame, output_dir: str, should_use_parallel: bool, rerun_data: bool, max_lag: int = 10, significance_level: float = 0.05) -> pd.DataFrame:
	"""
	Generate embeddings for each volume in the given DataFrame.

	Parameters:
	- volume_paths_df: DataFrame containing volume paths.
	- output_dir: Directory to save the embeddings.
	- should_use_parallel: Whether to use parallel processing.
	- rerun_data: Whether to rerun the data processing.

	Returns:
	- volume_frequencies: List of volume frequencies.
	"""
	console.print(f"[bright_cyan]Starting signal processing...Output dir{output_dir}, Should use Parallel? {should_use_parallel}, Should rerun data? {rerun_data}[/bright_cyan]")
	volume_frequencies = []
	volume_paths_df = volume_paths_df.reset_index(drop=True)
	volume_paths_df = volume_paths_df.sort_values(by=['table_row_index'])
	periodical_name = volume_paths_df['lowercase_periodical_name'].unique()[0]
	altair_charts = []
	for _, volume in volume_paths_df.iterrows():
		console.print(f"Processing volume: {volume['htid']}", style="bright_blue")
		merged_expanded_df, grouped_df, tokens_raw_signal, tokens_smoothed_signal = process_tokens(
			volume['file_path'], 
			volume['is_annotated_periodical'], 
			volume['should_filter_greater_than_numbers'], 
			volume['should_filter_implied_zeroes']
		)
		# Extract the directory path without the CSV file
		directory_path = os.path.dirname(volume['file_path'])

		# Create the new directory path for wavelet_analysis
		wavelet_analysis_dir = os.path.join(directory_path, 'wavelet_analysis')

		if rerun_data and os.path.exists(wavelet_analysis_dir):
			shutil.rmtree(wavelet_analysis_dir)

		# Create the wavelet_analysis directory if it doesn't exist
		os.makedirs(wavelet_analysis_dir, exist_ok=True)

		# Check stationarity for raw and smoothed signals
		raw_stationarity_result = check_wavelet_stationarity(tokens_raw_signal, signal_type="raw", max_lag=max_lag, significance_level=significance_level)
		smoothed_stationarity_result = check_wavelet_stationarity(tokens_smoothed_signal, signal_type="smoothed",max_lag=max_lag, significance_level=significance_level)
		wavelet_transform_settings = {
			'raw': {
				'is_stationary': raw_stationarity_result["is_stationary"],
				'original_signal': True
			},
			'smoothed': {
				'is_stationary': smoothed_stationarity_result["is_stationary"],
				'original_signal': True
			},
		}

		# Preprocess raw signal if non-stationary
		if not raw_stationarity_result["is_stationary"]:
			console.print("[yellow]Raw signal is not stationary. Attempting preprocessing...[/yellow]")
			tokens_raw_signal, raw_stationarity_result = preprocess_signal_for_stationarity(
				tokens_raw_signal, signal_type="raw", max_lag=max_lag, significance_level=significance_level
			)
			if not raw_stationarity_result["is_stationary"]:
				console.print("[red]Failed to preprocess raw signal. Skipping DWT/SWT for raw signal.[/red]")
				wavelet_transform_settings['raw']['original_signal'] = False
				wavelet_transform_settings['raw']['is_stationary'] = False

		# Preprocess smoothed signal if non-stationary
		if not smoothed_stationarity_result["is_stationary"]:
			console.print("[yellow]Smoothed signal is not stationary. Attempting preprocessing...[/yellow]")
			tokens_smoothed_signal, smoothed_stationarity_result = preprocess_signal_for_stationarity(
				tokens_smoothed_signal, signal_type="smoothed", max_lag=max_lag, significance_level=significance_level
			)
			if not smoothed_stationarity_result["is_stationary"]:
				console.print("[red]Failed to preprocess smoothed signal. Skipping DWT/SWT for smoothed signal.[/red]")
				wavelet_transform_settings['smoothed']['original_signal'] = False
				wavelet_transform_settings['smoothed']['is_stationary'] = False

		if tokens_raw_signal is None or tokens_smoothed_signal is None:
			console.print("[red]Skipping wavelet analysis due to non-stationary signals.[/red]")
			continue
		

		# Log stationarity results for both signals
		volume_data = {
			'raw_stationarity': raw_stationarity_result["is_stationary"],
			'raw_adf_pvalue': raw_stationarity_result.get("ADF p-value"),
			'raw_kpss_pvalue': raw_stationarity_result.get("KPSS p-value"),
			'smoothed_stationarity': smoothed_stationarity_result["is_stationary"],
			'smoothed_adf_pvalue': smoothed_stationarity_result.get("ADF p-value"),
			'smoothed_kpss_pvalue': smoothed_stationarity_result.get("KPSS p-value"),
		}
		signal_types = {
			"raw": tokens_raw_signal,
			"smoothed": tokens_smoothed_signal,
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
		signal_metrics_df.to_csv("test.csv", index=False)
		# Calculate wavelet metrics and signal metrics
		best_wavelet_config = compare_and_rank_wavelet_metrics(
			tokens_raw_signal, tokens_smoothed_signal, wavelet_analysis_dir, volume['htid'], signal_metrics_df, wavelet_transform_settings, should_use_parallel
		)
		
		

		# Separate raw and smoothed signals
		raw_signals = signal_metrics_df[signal_metrics_df.signal_type == 'raw'].drop(columns=['signal_type'])
		smoothed_signals = signal_metrics_df[signal_metrics_df.signal_type == 'smoothed'].drop(columns=['signal_type'])

		# Rename columns to include the signal_type
		raw_signals.columns = [f"raw_{col}" for col in raw_signals.columns]
		smoothed_signals.columns = [f"smoothed_{col}" for col in smoothed_signals.columns]

		# Concatenate the DataFrames side by side
		merged_signals = pd.concat([raw_signals.reset_index(drop=True), smoothed_signals.reset_index(drop=True)], axis=1)
		
		if volume['is_annotated_periodical'] and len(grouped_df) > 1:
			missing_issues, chart = plot_annotated_periodicals(merged_expanded_df, grouped_df, output_dir, volume['lowercase_periodical_name'], merged_signals.raw_dynamic_cutoff.values[0])
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
		volume_data.update({
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
		})
		
		# Merge the best_wavelet_dict with volume_data
		volume_data.update(best_wavelet_dict)
		volume_data.update(merged_signals_dict)
		# Append to the list of volume frequencies
		volume_frequencies.append(volume_data)
		volume_df = pd.DataFrame([volume_data])
		volume_df = volume_df.drop(columns=['tokens_per_page', 'page_numbers', 'digits_per_page'])


		wavelet_results_file_path = wavelet_analysis_dir + f"/{volume['htid'].replace('.', '_')}_wavelet_volume_results.csv"
		volume_df.to_csv(wavelet_results_file_path, index=False)

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

def generate_token_frequency_analysis(should_filter_greater_than_numbers: bool, should_filter_implied_zeroes: bool, only_use_annotated_periodicals: bool, load_existing_data: bool = False, rerun_analysis: bool = True, should_use_parallel: bool = False):
	"""
	Generates token frequency analysis for all identified HathiTrust periodicals.

	This function performs several tasks to analyze token frequency data extracted from periodicals:
	1. Identifies annotated periodicals and reads their metadata and token-level data.
	2. Checks for existing wavelet-processed data and loads it if `load_existing_data` is set to `True`.
	3. Iterates over a predefined list of preidentified periodicals title by title, processing their volumes either from scratch or by reusing previously saved data.
	4. Applies various filtering options (e.g., excluding numbers larger than the maximum page number or implied zeroes).
	5. Aggregates and processes volume-level data, generating signal processing features.
	6. Saves the resulting token frequency features to a consolidated CSV file for further analysis.

	Parameters:
	----------
	should_filter_greater_than_numbers: bool
		Flag indicating whether to filter out tokens representing numbers greater than the maximum possible page number. It is used in the function `process_tokens`, which passes it to the `process_file` function, and primarily for issue boundary detection.

	should_filter_implied_zeroes: bool
		Flag indicating whether to filter out tokens representing "implied zeroes." Implied zeroes are derived from the difference between a token digit and a page number. It is used in the function `process_tokens`, which passes it to the `process_file` function, and primarily for issue boundary detection.

	only_use_annotated_periodicals: bool
		Flag indicating whether to process only periodicals with manual annotations. If `True`, periodicals without annotation files are skipped to focus solely on validated datasets.

	load_existing_data: bool, default=False
		Whether to load pre-existing wavelet or token frequency analysis data if available. If `False`, the function deletes existing data and regenerates it from scratch.

	rerun_analysis: bool, default=True
		Whether to rerun the token frequency analysis for volumes. If `False`, existing data for already-processed volumes will not be overwritten.

	should_use_parallel: bool, default=False
		Whether to enable parallel processing for volume analysis. This can improve performance when processing large datasets by leveraging multiple CPU cores.

	Returns:
	--------
	None

	Notes:
	--------
	- The function assumes the existence of certain directories and files, such as the `HathiTrust-pcc-datasets` directory containing periodical metadata and token-level data. It does use the `get_data_directory_path` function to determine the correct path, as well as `os.path.join` to construct file paths so hopefully that will work regardless of the OS.

	"""
	# Count the number of matching files
	matching_files = []
	for directory, _, files in tqdm(os.walk(os.path.join("..", "..", "datasets", "annotated_ht_ef_datasets/")), desc="Counting matching files"):
		for file in files:
			if file.endswith(".csv") and 'individual' in file:
				if os.path.exists(os.path.join(directory, file)):
					publication_name = directory.split("/")[-2]
					volume_number = directory.split("/")[-1]
					matching_files.append({"file": file, "directory": directory, "file_path": os.path.join(directory, file), "periodical_title": publication_name, "volume_directory": volume_number})
	matching_files_df = pd.DataFrame(matching_files)
	console.print(f"Found {len(matching_files_df)} matching files.", style="bright_green")

	volume_features_output_path = os.path.join("..",  "..", "datasets", "all_volume_features_and_frequencies.csv")
	volume_features_exist = False
	if os.path.exists(volume_features_output_path) and load_existing_data:
		volume_features_df = read_csv_file(volume_features_output_path)
		volume_features_exist = True
		console.print(f"Found {len(volume_features_df)} existing volume features.", style="bright_green")
	elif os.path.exists(volume_features_output_path) and not load_existing_data:
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
				if (len(volume_in_features) > 0) and (not load_existing_data):
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
		volume_frequencies = generate_signal_processing_data(volume_paths_df, output_dir="../figures", should_use_parallel=should_use_parallel, rerun_data=rerun_analysis)
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
	should_rerun_code = True
	should_load_existing_data = False
	should_only_use_annotated_periodicals = True
	parallelization = True
	generate_token_frequency_analysis(filter_greater_than_numbers, filter_implied_zeroes,  should_only_use_annotated_periodicals, should_load_existing_data, should_rerun_code, parallelization)