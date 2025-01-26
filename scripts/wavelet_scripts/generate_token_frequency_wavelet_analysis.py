# Standard library imports
import os
import sys
import warnings
import shutil

# Third-party imports
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
from rich.console import Console
import pywt
from sklearn.preprocessing import RobustScaler

# Local application imports
sys.path.append("..")
from scripts.utils import read_csv_file, get_data_directory_path, save_chart, process_file, generate_table
from scripts.wavelet_scripts.generate_wavelet_signal_processing import evaluate_dwt_performance, evaluate_dwt_performance_parallel, evaluate_cwt_performance, evaluate_cwt_performance_parallel, evaluate_swt_performance, evaluate_swt_performance_parallel, calculate_signal_metrics

# Disable max rows for Altair
alt.data_transformers.disable_max_rows()

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

## DATA PROCESSING FUNCTIONS
def process_tokens(file_path: str, preidentified_periodical: bool, should_filter_greater_than_numbers: bool, should_filter_implied_zeroes: bool) -> tuple:
	"""
	Processes and cleans token-level data extracted from OCR-processed periodicals, preparing it for wavelet analysis.

	The function begins by reading token data from the specified file and applies the process_file function to expand tokens while filtering based on the provided flags. Metadata columns (e.g., enumeration_chronology, pub_date) are included by merging with a corresponding metadata CSV if available. For pre-identified periodicals, additional metadata fields such as start_issue and end_issue are included, and only relevant columns are retained.

	Token-level data is merged with digit counts to create a comprehensive DataFrame, with missing values filled as zeros for consistency. Token and digit signals are smoothed using a moving average (window size = 5) and then standardized by subtracting the mean and dividing by the standard deviation. As a sanity check, the first two rows of the processed DataFrame are printed using the generate_table utility. Finally, raw and smoothed token frequency signals are extracted as 1D arrays, ensuring no NaN or infinite values, and are prepared for further analysis.

	Core Assumptions:
	- Input files must include required columns (e.g., tokens_per_page, page_number), and a metadata CSV must exist for token data if columns like enumeration_chronology are missing.
	- Pre-identified periodicals provide additional metadata fields that the function incorporates.
	- The smoothing process (centered moving average, window size = 5) assumes relatively uniform page lengths for meaningful results.
	- Missing values are replaced with zeros to ensure compatibility with downstream tools.

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
		A tuple containing:
		- `merged_expanded_df` (pd.DataFrame): A DataFrame with token-level data, metadata, 
		  and processed signal columns, including smoothed and standardized token and digit signals.
		- `grouped_df` (pd.DataFrame): A grouped version of the expanded DataFrame, often 
		  useful for aggregating data by specific columns.
		- `tokens_raw_signal` (np.ndarray): The raw token frequency signal as a 1D array, 
		  ready for further analysis (e.g., FFT or wavelet transformations).
		- `tokens_smoothed_signal` (np.ndarray): The smoothed token frequency signal as a 1D 
		  array, obtained via a moving average.
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

def check_if_actual_issue(row: pd.Series, grouped_df: pd.DataFrame) -> bool:
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

def normalize_weights_dynamically(metrics: list, weights: dict, results_df: pd.DataFrame, ranking_config: dict, threshold:float=0.9, shared_weight_factor:float=0.7, specific_weight_factor:float=0.3) -> tuple:
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
		ranking_config["metrics"][metric]["was_shared"] = True

	for metric in specific_metrics:
		normalized_weights[metric] = (
			normalized_weights[metric] / specific_total_weight * specific_weight_factor
		)
		ranking_config["metrics"][metric]["was_specific"] = True

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

def determine_best_wavelet_representation(
	results_df: pd.DataFrame, signal_type: str, weights: dict = None, is_combined: bool = False, epsilon_threshold: float = 1e-6, penalty_weight: float = 0.05, percentage_of_results: float = 0.1
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
	console.print(f"Results for {signal_type} Wavelet Analysis")
	# Default weights if not provided
	if weights is None:
		weights = {
			'wavelet_mse': 0.2,
			'wavelet_psnr': 0.3,
			'wavelet_energy_entropy': 0.2,
			'wavelet_sparsity': 0.2,
			'wavelet_entropy': 0.1,
			'smoothness': 0.1,
			'correlation': 0.1,
			'avg_variance_across_levels': 0.1,
			'emd_value': 0.2,
			'kl_divergence': 0.2,
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
	metrics_to_remove = [
		metric for metric in weights
		if metric in results_df.columns and results_df[metric].std() <= epsilon_threshold
	]
	for metric in metrics_to_remove:
		avg_value = results_df[metric].mean()
		std_value = results_df[metric].std()
		ranking_config["metrics"][metric]["ignore_metric"] = True
		ranking_config["metrics"][metric]["removal_reason"] = f"Low variance (std: {std_value:.6f}, mean: {avg_value:.6f})"

		console.print(
			f"[yellow]Excluding '{metric}' from analysis due to low variance "
			f"(std: {std_dev:.6f}, mean: {avg_value:.6f}).[/yellow]"
		)

	# Log-transform extreme values in `wavelet_energy_entropy` if necessary
	if 'wavelet_energy_entropy' in results_df.columns:
		if results_df['wavelet_energy_entropy'].min() < 0:
			console.print(f"[yellow]Log-transforming `wavelet_energy_entropy` due to negative values.[/yellow]")
			results_df['wavelet_energy_entropy'] = np.log1p(np.abs(results_df['wavelet_energy_entropy']))
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
		if not metric_config.get("ignore_metric", False)  # Exclude metrics flagged with ignore_metric
		and metric_config["metric"] in results_df.columns  # Ensure the metric exists in the results_df
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
		metric_config["metric"] for metric_config in ranking_config["metrics"] if not metric_config.get("ignore_metric", False)
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
	use_parallel: bool = False,
	weights: dict = None,
) -> pd.DataFrame:
	"""
	Compare wavelet metrics for raw and smoothed tokens and determine the best representation.

	Parameters:
	-----------
	raw_signal : np.ndarray
		Raw tokens per page.
	smoothed_signal : np.ndarray
		Smoothed tokens per page.
	wavelet_directory : str
		Directory containing wavelet analysis results.
	volume_id : str
		Volume identifier for saving results.
	weights : dict, optional
		Weights for metrics (default uses standard weights for MSE, Energy-to-Entropy, and Sparsity).
	use_parallel : bool, optional
		Whether to use parallel processing for wavelet analysis.

	Returns:
	--------
	best_combined_results: pd.DataFrame
		  DataFrame containing the best combined wavelet configuration.
	"""
	if weights is None:
		weights = {
			'wavelet_mse': 0.3,
			'wavelet_psnr': 0.3,
			'wavelet_energy_entropy': 0.2,
			'wavelet_sparsity': 0.2,
			'wavelet_entropy': 0.1,
			'smoothness': 0.1,
			'correlation': 0.1,
			'avg_variance_across_levels': 0.1,
			'variance_ratio_across_levels': 0.1,
			'emd_value': 0.2,
			'kl_divergence': 0.2,
		}

	dwt_wavelets = pywt.wavelist(kind='discrete')
	cwt_wavelets = filter_wavelets(pywt.wavelist(kind='continuous'))
	modes = pywt.Modes.modes

	file_path = wavelet_directory + f"/{volume_id.replace('.', '_')}_"

	# Evaluate metrics for DWT
	try:
		if use_parallel:
			dwt_raw_results, dwt_raw_skipped_results = evaluate_dwt_performance_parallel(raw_signal, dwt_wavelets, modes, 'raw')
			dwt_smoothed_results, dwt_smoothed_skipped_results = evaluate_dwt_performance_parallel(smoothed_signal, dwt_wavelets, modes, 'smoothed')
		else:
			dwt_raw_results, dwt_raw_skipped_results = evaluate_dwt_performance(raw_signal, dwt_wavelets, modes, 'raw')
			dwt_smoothed_results, dwt_smoothed_skipped_results = evaluate_dwt_performance(smoothed_signal, dwt_wavelets, modes, 'smoothed')
	except Exception as e:
		console.print(f"[bright_red]Error evaluating DWT: {e}[/bright_red]")
		dwt_raw_results = dwt_smoothed_results = pd.DataFrame()
		dwt_raw_skipped_results = dwt_smoothed_skipped_results = pd.DataFrame()

	# Evaluate metrics for CWT
	try:
		if use_parallel:
			cwt_raw_results, cwt_raw_skipped_results = evaluate_cwt_performance_parallel(raw_signal, cwt_wavelets, 'raw')
			cwt_smoothed_results, cwt_smoothed_skipped_results = evaluate_cwt_performance_parallel(smoothed_signal, cwt_wavelets, 'smoothed')
		else:
			cwt_raw_results, cwt_raw_skipped_results = evaluate_cwt_performance(raw_signal, cwt_wavelets, 'raw')
			cwt_smoothed_results, cwt_smoothed_skipped_results = evaluate_cwt_performance(smoothed_signal, cwt_wavelets, 'smoothed')
	except Exception as e:
		console.print(f"[bright_red]Error evaluating CWT: {e}[/bright_red]")
		cwt_raw_results = cwt_smoothed_results = pd.DataFrame()
		cwt_raw_skipped_results = cwt_smoothed_skipped_results = pd.DataFrame()

	# Evaluate metrics for SWT
	try:
		if use_parallel:
			swt_raw_results, swt_raw_skipped_results = evaluate_swt_performance_parallel(raw_signal, dwt_wavelets, 'raw')
			swt_smoothed_results, swt_smoothed_skipped_results = evaluate_swt_performance_parallel(smoothed_signal, dwt_wavelets, 'smoothed')
		else:
			swt_raw_results, swt_raw_skipped_results = evaluate_swt_performance(raw_signal, dwt_wavelets, 'raw')
			swt_smoothed_results, swt_smoothed_skipped_results = evaluate_swt_performance(smoothed_signal, dwt_wavelets, 'smoothed')
	except Exception as e:
		console.print(f"[bright_red]Error evaluating SWT: {e}[/bright_red]")
		swt_raw_results = swt_smoothed_results = pd.DataFrame()
		swt_raw_skipped_results = swt_smoothed_skipped_results = pd.DataFrame()

	# Combine results
	dwt_combined_results = pd.concat([dwt_raw_results, dwt_smoothed_results], ignore_index=True)
	dwt_combined_skipped_results = pd.concat([dwt_raw_skipped_results, dwt_smoothed_skipped_results], ignore_index=True)
	if not dwt_combined_skipped_results.empty:
		dwt_combined_skipped_results.to_csv(f"{file_path}dwt_skipped_results.csv", index=False)

	cwt_combined_results = pd.concat([cwt_raw_results, cwt_smoothed_results], ignore_index=True)
	cwt_combined_skipped_results = pd.concat([cwt_raw_skipped_results, cwt_smoothed_skipped_results], ignore_index=True)
	if not cwt_combined_skipped_results.empty:
		cwt_combined_skipped_results.to_csv(f"{file_path}cwt_skipped_results.csv", index=False)

	swt_combined_results = pd.concat([swt_raw_results, swt_smoothed_results], ignore_index=True)
	swt_combined_skipped_results = pd.concat([swt_raw_skipped_results, swt_smoothed_skipped_results], ignore_index=True)
	if not swt_combined_skipped_results.empty:
		swt_combined_skipped_results.to_csv(f"{file_path}swt_skipped_results.csv", index=False)

	# Ensure results are non-empty before ranking
	table_cols = ['wavelet_rank', 'final_wavelet_rank', 'final_score', 'wavelet_norm_weighted_score', 'normalized_diff', 'wavelet_zscore_weighted_score', 'missing_metrics_count']
	if not dwt_combined_results.empty:
		best_dwt, ranked_dwt, subset_ranked_dwt, dwt_correlation_score = determine_best_wavelet_representation(dwt_combined_results, "DWT", weights, False)
		subset_ranked_dwt['wavelet_type'] = 'DWT'
		ranked_dwt.to_csv(f"{file_path}full_dwt_results.csv", index=False)
		subset_ranked_dwt.to_csv(f"{file_path}subset_dwt_results.csv", index=False)
		if not best_dwt.empty:
			generate_table(best_dwt[ ['wavelet', 'signal_type'] + table_cols], f"Best DWT Wavelet Configuration (Correlation: {dwt_correlation_score:.2f})")
	else:
		subset_ranked_dwt = pd.DataFrame()

	if not cwt_combined_results.empty:
		best_cwt, ranked_cwt, subset_ranked_cwt, cwt_correlation_score = determine_best_wavelet_representation(cwt_combined_results, "CWT", weights, False)
		subset_ranked_cwt['wavelet_type'] = 'CWT'
		ranked_cwt.to_csv(f"{file_path}full_cwt_results.csv", index=False)
		subset_ranked_cwt.to_csv(f"{file_path}subset_cwt_results.csv", index=False)
		if not best_cwt.empty:
			generate_table(best_cwt[['wavelet', 'signal_type'] + table_cols], f"Best CWT Wavelet Configuration (Correlation: {cwt_correlation_score:.2f})")
	else:
		subset_ranked_cwt = pd.DataFrame()

	if not swt_combined_results.empty:
		best_swt, ranked_swt, subset_ranked_swt, swt_correlation_score = determine_best_wavelet_representation(swt_combined_results, "SWT", weights, False)
		subset_ranked_swt['wavelet_type'] = 'SWT'
		ranked_swt.to_csv(f"{file_path}full_swt_results.csv", index=False)
		subset_ranked_swt.to_csv(f"{file_path}subset_swt_results.csv", index=False)
		if not best_swt.empty:
			generate_table(best_swt[['wavelet', 'signal_type'] + table_cols], f"Best SWT Wavelet Configuration (Correlation: {swt_correlation_score:.2f})")
	else:
		subset_ranked_swt = pd.DataFrame()

	# Combine DWT, CWT, and SWT results
	combined_results = pd.concat([subset_ranked_dwt, subset_ranked_cwt, subset_ranked_swt], ignore_index=True)
	table_cols = ['combined_' + col for col in table_cols]
	# Determine overall best representation
	if not combined_results.empty:
		best_combined_results, ranked_combined_results, subset_ranked_combined_results, combined_correlation_score = determine_best_wavelet_representation(
			combined_results, "Combined", weights, True
		)
		ranked_combined_results.to_csv(f"{file_path}combined_results.csv", index=False)
		subset_ranked_combined_results.to_csv(f"{file_path}subset_combined_results.csv", index=False)
		if not best_combined_results.empty:
			generate_table(best_combined_results[['wavelet', 'signal_type', 'wavelet_type'] + table_cols], f"Best Combined Wavelet Configuration (Correlation: {combined_correlation_score:.2f})")
	else:
		best_combined_results = None

	return best_combined_results

## PLOTTING FUNCTIONS

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

def plot_annotated_periodicals(merged_expanded_df, grouped_df, output_dir, periodical_name, dynamic_cutoff):
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

## MAIN FUNCTIONS

def generate_signal_processing_data_parallel(volume_paths_df: pd.DataFrame, output_dir: str, should_use_parallel: bool) -> pd.DataFrame:
	"""
	Parallelized generation of signal processing data for multiple volumes.

	Parameters:
	-----------
	volume_paths_df : pd.DataFrame
		DataFrame containing volume metadata and file paths.
	output_dir : str
		Directory to save output files and results.

	Returns:
	--------
	pd.DataFrame
		DataFrame with aggregated results for all processed volumes.
	"""
	def process_volume(volume):
		"""
		Process a single volume for wavelet and signal analysis.
		"""
		try:
			# Process tokens and signals
			merged_expanded_df, grouped_df, tokens_raw_signal, tokens_smoothed_signal = process_tokens(
				volume['file_path'],
				volume['is_annotated_periodical'],
				volume['should_filter_greater_than_numbers'],
				volume['should_filter_implied_zeroes'],
			)

			# Perform wavelet analysis
			wavelet_results_df, best_wavelet_config, combined_wavelet_correlation, dwt_skipped, cwt_skipped, swt_skipped = compare_and_rank_wavelet_metrics(
				tokens_raw_signal, tokens_smoothed_signal, should_use_parallel
			)

			# Calculate signal metrics
			signal_metrics_results = []
			for signal_type, signal in {
				"raw": merged_expanded_df['tokens_per_page'].values,
				"smoothed": merged_expanded_df['smoothed_tokens_per_page'].values,
			}.items():
				result = calculate_signal_metrics(
					tokens_signal=signal,
					use_signal_type=signal_type,
					min_tokens=merged_expanded_df['tokens_per_page'].min(),
					prominence=1.0,
					distance=5,
					verbose=False,
				)
				signal_metrics_results.append(result)
			signal_metrics_df = pd.DataFrame(signal_metrics_results)

			# Annotate and plot issues for annotated periodicals
			missing_issues, chart = [], None
			if volume['is_annotated_periodical'] and len(grouped_df) > 1:
				missing_issues, chart = plot_annotated_periodicals(
					merged_expanded_df,
					grouped_df,
					output_dir,
					volume['lowercase_periodical_name'],
					signal_metrics_df.iloc[0].get('raw_dynamic_cutoff', 0),
				)

			# Collect results
			best_wavelet_dict = best_wavelet_config.iloc[0].to_dict()
			return {
				"htid": merged_expanded_df['htid'].unique()[0],
				"lowercase_periodical_name": volume['lowercase_periodical_name'],
				"avg_tokens": merged_expanded_df['tokens_per_page'].mean(),
				"avg_digits": merged_expanded_df['digits_per_page'].mean(),
				"raw_likely_covers": merged_expanded_df[merged_expanded_df['tokens_per_page'] <= signal_metrics_df.iloc[0].get('raw_dynamic_cutoff', 0)].page_number.tolist(),
				"wavelet_correlation": combined_wavelet_correlation,
				"wavelet_results_df": wavelet_results_df,
				"dwt_skipped": dwt_skipped,
				"cwt_skipped": cwt_skipped,
				"swt_skipped": swt_skipped,
				"missing_issues": missing_issues,
				"chart": chart,
			}
		except Exception as e:
			return {"error": str(e), "volume": volume}

	# Prepare for parallel processing
	volume_paths = volume_paths_df.to_dict(orient="records")
	results = []
	max_workers = min(len(volume_paths), multiprocessing.cpu_count() - 1)
	console.print(f"[cyan]Using {max_workers} workers for parallel processing.[/cyan]")

	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(process_volume, volume) for volume in volume_paths]
		for future in tqdm(as_completed(futures), total=len(futures), desc="Processing volumes"):
			try:
				result = future.result()
				if "error" not in result:
					results.append(result)
				else:
					console.print(f"[red]Error in volume: {result['error']}[/red]")
			except Exception as e:
				console.print(f"[red]Unexpected error during volume processing: {e}[/red]")

	# Aggregate results
	processed_volumes = pd.DataFrame(results)
	console.print(f"[green]Successfully processed {len(processed_volumes)} volumes.[/green]")

	# Handle Altair charts
	altair_charts = [result["chart"] for result in results if result.get("chart")]
	if altair_charts:
		combined_charts = alt.vconcat(*altair_charts)
		save_chart(
			combined_charts, f"{output_dir}/annotated_tokens_per_page/{volume_paths_df['lowercase_periodical_name'].iloc[0]}_tokens_per_page_chart.png", scale_factor=2.0
		)

	# Save to CSV
	processed_volumes.to_csv(os.path.join(output_dir, "processed_volumes.csv"), index=False)

	return processed_volumes

def generate_signal_processing_data(volume_paths_df: pd.DataFrame, output_dir: str, should_use_parallel: bool, rerun_data: bool) -> pd.DataFrame:
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
		# Calculate wavelet metrics and signal metrics
		best_wavelet_config = compare_and_rank_wavelet_metrics(
			tokens_raw_signal, tokens_smoothed_signal, wavelet_analysis_dir, volume['htid'], should_use_parallel
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
		volume_data = {
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
		
		# Merge the best_wavelet_dict with volume_data
		volume_data.update(best_wavelet_dict)
		volume_data.update(merged_signals_dict)
		# Append to the list of volume frequencies
		volume_frequencies.append(volume_data)
		volume_df = pd.DataFrame([volume_data])
		volume_df = volume_df.drop(columns=['tokens_per_page', 'page_numbers', 'digits_per_page'])
		# merged_volume_df 
		# # Merge the full wavelet results with the signal metrics
		# merged_signal_analysis_df = pd.merge(wavelet_results_df, signal_metrics_df, on='signal_type', how='left')
		# # Drop the positive frequencies and amplitudes
		# merged_signal_analysis_df = merged_signal_analysis_df.drop(columns=['positive_frequencies', 'positive_amplitudes'])
		# # Concat the full wavelet results to the volume data and store as CSV
		# repeated_volume_df = pd.concat([volume_df] * len(merged_signal_analysis_df), ignore_index=True)

		# full_wavelet_results = pd.concat([repeated_volume_df, merged_signal_analysis_df], axis=1)

		

		wavelet_results_file_path = wavelet_analysis_dir + f"/{volume['htid'].replace('.', '_')}_wavelet_volume_results.csv"
		volume_df.to_csv(wavelet_results_file_path, index=False)

		# skipped_dwts_file_path = wavelet_analysis_dir + f"/{volume['htid']}_skipped_dwts.csv"
		# if len(dwt_skipped_results) > 0:
		# 	dwt_skipped_results.to_csv(skipped_dwts_file_path, index=False)

		# skipped_cwts_file_path = wavelet_analysis_dir + f"/{volume['htid']}_skipped_cwts.csv"
		# if len(cwt_skipped_results) > 0:
		# 	cwt_skipped_results.to_csv(skipped_cwts_file_path, index=False)

		# skipped_swts_file_path = wavelet_analysis_dir + f"/{volume['htid']}_skipped_swts.csv"
		# if len(swt_skipped_results) > 0:
		# 	swt_skipped_results.to_csv(skipped_swts_file_path, index=False)


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
	should_only_use_annotated_periodicals = False
	parallelization = True
	generate_token_frequency_analysis(filter_greater_than_numbers, filter_implied_zeroes,  should_only_use_annotated_periodicals, should_load_existing_data, should_rerun_code, parallelization)