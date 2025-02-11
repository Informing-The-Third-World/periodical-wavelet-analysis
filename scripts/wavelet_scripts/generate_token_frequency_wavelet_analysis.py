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

# Local application imports
sys.path.append("../..")
from scripts.utils import read_csv_file, get_data_directory_path, save_chart, generate_table, process_tokens
from scripts.wavelet_scripts.generate_wavelet_stationarity import preprocess_signal_for_stationarity, check_wavelet_stationarity
from scripts.wavelet_scripts.generate_wavelet_signal_processing import evaluate_dwt_performance, evaluate_dwt_performance_parallel, evaluate_cwt_performance, evaluate_cwt_performance_parallel, evaluate_swt_performance, evaluate_swt_performance_parallel, calculate_signal_metrics
from scripts.wavelet_scripts.generate_wavelet_plots import plot_volume_frequencies_matplotlib, plot_tokens_per_page, plot_annotated_periodicals
from scripts.wavelet_scripts.generate_wavelet_rankings import determine_best_wavelet_representation

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

def convert_to_native_types(obj):
    """
    Convert numpy data types to native Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

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

	for wavelet_type, wavelet_info in wavelet_types.items():
		console.print(f"[blue]Processing {wavelet_type} wavelet type[/blue]")
		wavelet_results = []
		individual_wavelet_directory = os.path.join(wavelet_directory, f"{wavelet_type}_results")
		os.makedirs(individual_wavelet_directory, exist_ok=True)
		for signal_type, settings in wavelet_transform_settings.items():
			signal = raw_signal if signal_type == 'raw' else smoothed_signal
			console.print(f"[blue]  Processing {signal_type} signal for {wavelet_type}[/blue]")
			# Skip DWT and SWT for non-stationary signals
			if wavelet_type in ["DWT", "SWT"] and not settings["is_stationary"]:
				console.print(f"[yellow]  Skipping {wavelet_type} for {signal_type} signal (non-stationary).[/yellow]")
				continue
			# Process the wavelet type
			results, skipped_results = process_wavelet_type(wavelet_type, wavelet_info, signal, modes, signal_type)
			individual_signal_directory = os.path.join(individual_wavelet_directory, f"{signal_type}_results")
			os.makedirs(individual_signal_directory, exist_ok=True)
			individual_signal_file_path = individual_signal_directory + f"/{volume_id.replace('.', '_')}_"
			if not results.empty:
				results['wavelet_type'] = wavelet_type
				results['signal_type'] = signal_type
				results['htid'] = volume_id
				subset_ranked, ranked, ranking_config = determine_best_wavelet_representation(
					results, signal_type, signal_metrics_df[signal_metrics_df.signal_type == signal_type]
				)

				suffix = f"{wavelet_type.lower()}_{signal_type}"
				ranked.to_csv(f"{individual_signal_file_path}full_ranked_results.csv", index=False)
				subset_ranked.to_csv(f"{individual_signal_file_path}subset_ranked_results.csv", index=False)
				ranking_config = json.loads(json.dumps(ranking_config, default=convert_to_native_types))
				with open(f"{individual_signal_file_path}ranking_config.json", "w") as f:
					json.dump(ranking_config, f, indent=4)
				
				wavelet_results.append(ranked)
			if not skipped_results.empty:
				skipped_results['wavelet_type'] = wavelet_type
				skipped_results['signal_type'] = signal_type
				skipped_results.to_csv(f"{individual_signal_file_path}skipped_results.csv", index=False)
				
		# Save skipped results
		results_df = pd.concat(wavelet_results, ignore_index=True)
		
		print(f"Results {len(results_df)} for {wavelet_type} Wavelet Type")
		individual_wavelet_file_path = wavelet_directory + f"/{volume_id.replace('.', '_')}_"
		# Save results and rank them
		if not results_df.empty:
			ranked, subset_ranked, ranking_config = determine_best_wavelet_representation(
				results_df, wavelet_type, signal_metrics_df, weights, False
			)
			ranked.to_csv(f"{individual_wavelet_file_path}full_ranked_results.csv", index=False)
			subset_ranked.to_csv(f"{individual_wavelet_file_path}subset_ranked_results.csv", index=False)
			ranking_config = json.loads(json.dumps(ranking_config, default=convert_to_native_types))
			with open(f"{individual_wavelet_file_path}ranking_config.json", "w") as f:
				json.dump(ranking_config, f, indent=4)

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
		combined_config = json.loads(json.dumps(combined_config, default=convert_to_native_types))
		with open(f"{file_path}combined_{suffix}ranking_config.json", "w") as f:
			json.dump(combined_config, f, indent=4)
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
		signal_volume_data = pd.DataFrame([volume_data])
		signal_metrics_df = pd.concat([signal_volume_data, signal_metrics_df], axis=1)
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