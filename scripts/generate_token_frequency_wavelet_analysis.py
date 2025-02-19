# Standard library imports
import os
import sys
import warnings
import shutil
import json
from typing import Tuple

# Third-party imports
import pandas as pd
import numpy as np
import altair as alt
from tqdm import tqdm
from rich.console import Console
import pywt

# Local application imports
sys.path.append("..")
from scripts.utils import read_csv_file, get_data_directory_path, save_chart, process_tokens
from scripts.generate_wavelet_stationarity import preprocess_signal_for_stationarity, check_wavelet_stationarity
from scripts.generate_wavelet_signal_processing import evaluate_dwt_performance, evaluate_dwt_performance_parallel, evaluate_cwt_performance, evaluate_cwt_performance_parallel, evaluate_swt_performance, evaluate_swt_performance_parallel, calculate_signal_metrics
from scripts.generate_wavelet_plots import plot_volume_frequencies_matplotlib, plot_tokens_per_page, plot_annotated_periodicals
from scripts.generate_wavelet_rankings import determine_best_wavelet_representation

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize console
console = Console()

## WAVELET SIGNAL PROCESSING FUNCTIONS
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

def ensure_stationarity_for_signals(
	tokens_raw_signal: np.ndarray,
	tokens_smoothed_signal: np.ndarray,
	max_lag: int,
	significance_level: float
) -> Tuple[np.ndarray, np.ndarray, dict, bool, dict]:
	"""
	Checks stationarity for raw and smoothed signals. If non-stationary, attempts preprocessing. If preprocessing fails, skips wavelet analysis.

	Parameters
	----------
	tokens_raw_signal : np.ndarray
		1D array representing the raw signal.
	tokens_smoothed_signal : np.ndarray
		1D array representing the smoothed signal.
	max_lag : int
		Maximum lag for stationarity tests.
	significance_level : float
		Significance level for stationarity tests.

	Returns
	-------
	tokens_raw_signal : np.ndarray
		Preprocessed raw signal.
	tokens_smoothed_signal : np.ndarray
		Preprocessed smoothed signal.
	wavelet_transform_settings : dict
		Settings for wavelet transforms.
	skip_analysis : bool
		Whether to skip wavelet analysis.
	signal_data : dict
		Stationarity results for raw and smoothed signals.
	"""

	console.print("[cyan]Checking stationarity for raw and smoothed signals...[/cyan]")

	# 1. Check stationarity on raw signal
	raw_stationarity_result = check_wavelet_stationarity(
		tokens_raw_signal,
		signal_type="raw",
		max_lag=max_lag,
		significance_level=significance_level
	)
	# 2. Check stationarity on smoothed signal
	smoothed_stationarity_result = check_wavelet_stationarity(
		tokens_smoothed_signal,
		signal_type="smoothed",
		max_lag=max_lag,
		significance_level=significance_level
	)

	# wavelet_transform_settings
	wavelet_transform_settings = {
		"raw": {
			"is_stationary": raw_stationarity_result["is_stationary"],
			"original_signal": True
		},
		"smoothed": {
			"is_stationary": smoothed_stationarity_result["is_stationary"],
			"original_signal": True
		},
	}

	skip_analysis = False  # We'll set this True if we can't proceed

	# If raw not stationary -> attempt preprocess
	if not raw_stationarity_result["is_stationary"]:
		console.print("[yellow]Raw signal is not stationary. Attempting preprocessing...[/yellow]")
		if tokens_raw_signal is not None:
			tokens_raw_signal, raw_stationarity_result = preprocess_signal_for_stationarity(
				tokens_raw_signal, signal_type="raw", max_lag=max_lag, significance_level=significance_level
			)
		if tokens_raw_signal is None or not raw_stationarity_result["is_stationary"]:
			console.print("[red]Failed to preprocess raw signal. Skipping DWT/SWT for raw signal.[/red]")
			wavelet_transform_settings["raw"]["is_stationary"] = False
			wavelet_transform_settings["raw"]["original_signal"] = False

	# If smoothed not stationary -> attempt preprocess
	if not smoothed_stationarity_result["is_stationary"]:
		console.print("[yellow]Smoothed signal is not stationary. Attempting preprocessing...[/yellow]")
		if tokens_smoothed_signal is not None:
			tokens_smoothed_signal, smoothed_stationarity_result = preprocess_signal_for_stationarity(
				tokens_smoothed_signal, signal_type="smoothed", max_lag=max_lag, significance_level=significance_level
			)
		if tokens_smoothed_signal is None or not smoothed_stationarity_result["is_stationary"]:
			console.print("[red]Failed to preprocess smoothed signal. Skipping DWT/SWT for smoothed signal.[/red]")
			wavelet_transform_settings["smoothed"]["is_stationary"] = False
			wavelet_transform_settings["smoothed"]["original_signal"] = False

	# If both raw and smoothed ended up None or not original, do we skip the entire wavelet analysis?
	if tokens_raw_signal is None and tokens_smoothed_signal is None:
		skip_analysis = True

	signal_data = {
		'raw_stationarity': raw_stationarity_result["is_stationary"],
		'raw_adf_pvalue': raw_stationarity_result.get("ADF p-value"),
		'raw_kpss_pvalue': raw_stationarity_result.get("KPSS p-value"),
		'smoothed_stationarity': smoothed_stationarity_result["is_stationary"],
		'smoothed_adf_pvalue': smoothed_stationarity_result.get("ADF p-value"),
		'smoothed_kpss_pvalue': smoothed_stationarity_result.get("KPSS p-value"),
	}
	console.print("[cyan]Completed stationarity checks and preprocessing.[/cyan]")
	return tokens_raw_signal, tokens_smoothed_signal, wavelet_transform_settings, skip_analysis, signal_data

def compute_signal_metrics_for_raw_and_smoothed(
    tokens_raw_signal: np.ndarray,
    tokens_smoothed_signal: np.ndarray,
    merged_expanded_df: pd.DataFrame,
    signal_data: dict,
    prominence: float = 1.0,
    distance: int = 5,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Computes metrics for both 'raw' and 'smoothed' signals, then returns a combined DataFrame.

    Parameters
    ----------
    tokens_raw_signal : np.ndarray
        The raw token-based signal.
    tokens_smoothed_signal : np.ndarray
        The smoothed token-based signal.
    merged_expanded_df : pd.DataFrame
        DataFrame containing page-level data from which we can derive min_tokens or other fields.
    signal_data : dict
        A dictionary of metadata (e.g., volume data) to include in the final DataFrame.
    prominence : float, optional
        Used by calculate_signal_metrics.
    distance : int, optional
        Used by calculate_signal_metrics.
    verbose : bool, optional
        Whether to print debugging info in calculate_signal_metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame combining metadata from signal_data plus the metrics for raw and smoothed signals.
        Typically has two rows (one for raw, one for smoothed).
    """

    # We'll map signal_type -> actual signal
    signal_types = {
        "raw": tokens_raw_signal,
        "smoothed": tokens_smoothed_signal,
    }
    signal_metrics_results = []

    # For raw and smoothed, compute metrics
    for signal_type, signal in signal_types.items():
        if signal is None:
            # If signal is None, skip to avoid errors
            continue

        # Calculate the minimum tokens from merged_expanded_df if needed
        min_tokens = merged_expanded_df['tokens_per_page'].min() if 'tokens_per_page' in merged_expanded_df.columns else 1

        # Call your existing metric function
        result = calculate_signal_metrics(
            tokens_signal=signal,
            use_signal_type=signal_type,
            min_tokens=min_tokens,
            prominence=prominence,
            distance=distance,
            verbose=verbose
        )
        # Append result for each signal (raw or smoothed)
        signal_metrics_results.append(result)

    # Convert to DataFrame
    signal_metrics_df = pd.DataFrame(signal_metrics_results)

    # Also convert 'signal_data' dict to a one-row DataFrame for merging
    signal_data_df = pd.DataFrame([signal_data])

    # Combine them horizontally
    combined_metrics_df = pd.concat([signal_data_df, signal_metrics_df], axis=1)
    return combined_metrics_df

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

def process_wavelet_type(wavelet_type: str, wavelet_info: dict, signal: np.ndarray, modes: list, signal_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Process a given wavelet type for a signal and return results.

	Parameters
	----------
	wavelet_type : str
		The wavelet type to process (e.g., DWT, CWT, SWT).
	wavelet_info : dict
		Dictionary containing wavelet information (wavelets, evaluation function).
	signal : np.ndarray
		The signal to process.
	modes : list
		List of PyWavelets modes.
	signal_type : str
		The type of signal (raw or smoothed).
	
	Returns
	-------
	pd.DataFrame
		DataFrame containing results for the wavelet type.
	pd.DataFrame
		DataFrame containing skipped results for the wavelet type.
	"""
	console.print(f"[blue]Processing {wavelet_type} wavelets for {signal_type} signal...[/blue]")
	# Evaluate the wavelet type. If DWT, pass modes.
	if wavelet_type == "DWT":
		results, skipped_results = wavelet_info["evaluate"](signal, wavelet_info["wavelets"], modes, signal_type)
	else:
		results, skipped_results = wavelet_info["evaluate"](signal, wavelet_info["wavelets"], signal_type)
	return results, skipped_results

def save_wavelet_results(file_path: str, results: pd.DataFrame, config: dict = None, suffix: str = "") -> None:
    """
	Helper function to save results and configuration files.
	
	Parameters
	----------
	file_path : str
		Path to the file to save.
	results : pd.DataFrame
		DataFrame containing results to save.
	config : dict, optional
		Dictionary containing configuration settings.
	suffix : str, optional
		Suffix to append to the file name.
	
	Returns
	-------
	None
	"""
    if not results.empty:
        results.to_csv(f"{file_path}_{suffix}.csv", index=False)
        if config:
            config = json.loads(json.dumps(config, default=convert_to_native_types))
            with open(f"{file_path}_{suffix}_config.json", "w") as f:
                json.dump(config, f, indent=4)
    else:
        console.print("[red]No valid wavelet configurations found.[/red]")

def process_signal_results(results_df: pd.DataFrame,signal_metrics_df: pd.DataFrame,file_path: str,context: str,prefix: str = "",**ranking_params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
	Process and save results for a given context (signal type or wavelet type).
	
	Parameters
	----------
	results_df : pd.DataFrame
		DataFrame containing results to process.
	signal_metrics_df : pd.DataFrame
		DataFrame containing signal metrics.
	file_path : str
		Path to the file to save.
	context : str
		Context for the results (signal type or wavelet type).
	prefix : str, optional
		Prefix to add to certain metric columns.
	ranking_params : dict
		Dictionary containing ranking parameters.
		
	Returns
	-------
	pd.DataFrame
		DataFrame containing subset of top wavelet configurations.
	pd.DataFrame
		DataFrame containing ranked wavelet configurations.
	"""
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    console.print(f"Calculating best wavelet representation for {context}...", style="bright_cyan")
    
    subset_ranked, ranked, ranking_config = determine_best_wavelet_representation(
        results_df,
        context,
        signal_metrics_df,
        prefix=prefix,
        **ranking_params
    )
    
    save_wavelet_results(file_path, ranked, ranking_config, "full_ranked_results")
    save_wavelet_results(file_path, subset_ranked, None, "subset_ranked_results")
    
    return subset_ranked, ranked

def compare_and_rank_wavelet_metrics(
    raw_signal: np.ndarray,
    smoothed_signal: np.ndarray,
    wavelet_directory: str,
    volume_id: str,
    signal_metrics_df: pd.DataFrame,
    wavelet_transform_settings: dict,
    use_parallel: bool = True,
    epsilon_threshold: float = 1e-6,
    penalty_weight: float = 0.05,
    percentage_of_results: float = 0.1,
    ignore_low_variance: bool = True
) -> pd.DataFrame:
    """
	This function evaluates wavelet performance for a raw signal and its smoothed counterpart across multiple wavelet transforms (DWT, CWT, SWT), then ranks and combines configurations to determine the best overall wavelet representation for the signal.

	Parameters
	----------
	raw_signal : np.ndarray
		The original (raw) 1D signal.
	smoothed_signal : np.ndarray
		The smoothed version of the signal.
	wavelet_directory : str
		Directory to save intermediate and final results.
	volume_id : str
		Identifying string (e.g. a volume name or ID).
	signal_metrics_df : pd.DataFrame
		DataFrame containing “original” signal metrics for raw and smoothed signals
		(one row per signal type, used by determine_best_wavelet_representation).
	wavelet_transform_settings : dict
		Dict specifying if signals are stationary or not, or other transform parameters.
	use_parallel : bool
		Whether to run wavelet evaluations in parallel.
	prefix : str
		Prefix for certain metric columns (if needed).
	epsilon_threshold : float
		Threshold for near-zero variance or presence checks in the pipeline.
	penalty_weight : float
		Weight factor for penalizing missing metrics in the final dynamic score.
	percentage_of_results : float
		Fraction of top results to select for the final subset (default is 0.1).
	ignore_low_variance : bool
		If True, skip metrics with near-zero variance in the first pass.

	Returns
	-------
	pd.DataFrame
		DataFrame containing top wavelet configurations across all wavelet types and signals.
	"""
    
    modes = pywt.Modes.modes
    ranking_params = {
        'epsilon_threshold': epsilon_threshold,
        'penalty_weight': penalty_weight,
        'percentage_of_results': percentage_of_results,
        'ignore_low_variance': ignore_low_variance
    }

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

    all_results = []
    subset_results = []

    # Process each wavelet type
    for wavelet_type, wavelet_info in wavelet_types.items():
        console.print(f"=== Processing wavelet type: {wavelet_type} ===", style="royal_blue1")
        wavelet_results = []
        individual_wavelet_directory = os.path.join(wavelet_directory, f"{wavelet_type}_results")
        os.makedirs(individual_wavelet_directory, exist_ok=True)

        # Process each signal type
        for signal_type, settings in wavelet_transform_settings.items():
            signal = raw_signal if signal_type == 'raw' else smoothed_signal
            console.print(f"For {signal_type} signal...", style="light_steel_blue")
            
            # Validate signal metrics
            subset_signal_metrics_df = signal_metrics_df[signal_metrics_df.signal_type == signal_type].reset_index(drop=True)
            if len(subset_signal_metrics_df) != 1:
                console.print(f"[red]Invalid number of signal metrics for {signal_type} signal. Skipping ranking...[/red]")
                if len(subset_signal_metrics_df) > 1:
                    subset_signal_metrics_df.to_csv("too_many_signal_metrics.csv", index=False)
                continue

            # Skip non-stationary signals for DWT/SWT
            if wavelet_type in ["DWT", "SWT"] and not settings.get("is_stationary", True):
                console.print(f"[yellow]  Skipping {wavelet_type} for {signal_type} signal (non-stationary).[/yellow]")
                continue

            individual_signal_directory = os.path.join(individual_wavelet_directory, f"{signal_type}_results")
            os.makedirs(individual_signal_directory, exist_ok=True)
            individual_signal_file_path = os.path.join(individual_signal_directory, f"{volume_id.replace('.', '_')}")

            # Load existing results or process new ones
            if os.path.exists(f"{individual_signal_file_path}_full_ranked_results.csv"):
                console.print(f"Results found for {wavelet_type} {signal_type} signal. Loading...", style="dark_sea_green2")
                ranked = pd.read_csv(f"{individual_signal_file_path}_full_ranked_results.csv")
                wavelet_results.append(ranked)
                continue

            results, skipped_results = process_wavelet_type(wavelet_type, wavelet_info, signal, modes, signal_type)
            
            if not results.empty:
                results.update({
                    'wavelet_type': wavelet_type,
                    'signal_type': signal_type,
                    'htid': volume_id
                })
                
                subset_ranked, ranked = process_signal_results(
                    results,
                    subset_signal_metrics_df,
                    individual_signal_file_path,
                    signal_type,
                    **ranking_params
                )
                wavelet_results.append(ranked)

            # Save skipped results
            if not skipped_results.empty:
                skipped_results.update({
                    'wavelet_type': wavelet_type,
                    'signal_type': signal_type,
                    'htid': volume_id
                })
                skipped_results.to_csv(f"{individual_signal_file_path}_skipped_results.csv", index=False)

        # Process results for this wavelet type
        results_df = pd.concat(wavelet_results, ignore_index=True) if wavelet_results else pd.DataFrame()
        console.print(f"Results found: {len(results_df)} rows for {wavelet_type} wavelet type.", style="dark_sea_green2")

        if not results_df.empty:
            individual_wavelet_file_path = os.path.join(individual_wavelet_directory, f"{volume_id.replace('.', '_')}")
            subset_ranked, ranked = process_signal_results(
                results_df,
                signal_metrics_df,
                individual_wavelet_file_path,
                wavelet_type,
                prefix="across_",
                **ranking_params
            )
            
            subset_ranked['wavelet_type'] = wavelet_type
            ranked['wavelet_type'] = wavelet_type
            subset_results.append(subset_ranked)
            all_results.append(ranked)

    # Process combined results
    individual_combined_file_path = os.path.join(wavelet_directory, f"{volume_id.replace('.', '_')}")
    
    combined_all_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined_subset_results = pd.concat(subset_results, ignore_index=True) if subset_results else pd.DataFrame()
    
    # Process and save final combined results
    subset_all_combined, _ = process_signal_results(
        combined_all_results,
        signal_metrics_df,
        individual_combined_file_path,
        "Combined",
        prefix="combined_",
        suffix="all",
        **ranking_params
    )
    
    subset_combined, _ = process_signal_results(
        combined_subset_results,
        signal_metrics_df,
        individual_combined_file_path,
        "Combined",
        prefix="combined_",
        **ranking_params
    )

    return subset_all_combined[0:1]

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

		split_directory_path = directory_path.split('datasets/')[-2:]
		wavelet_analysis_dir = os.path.join("..", "datasets", split_directory_path[0] + "datasets/", split_directory_path[1])

		if rerun_data and os.path.exists(wavelet_analysis_dir):
			shutil.rmtree(wavelet_analysis_dir)

		# Create the wavelet_analysis directory if it doesn't exist
		os.makedirs(wavelet_analysis_dir, exist_ok=True)
		console.print(f"Wavelet analysis directory: {wavelet_analysis_dir}", style="chartreuse1")
		# Ensure stationarity for signals
		tokens_raw_signal, tokens_smoothed_signal, wavelet_transform_settings, skip_analysis, signal_data = ensure_stationarity_for_signals(
			tokens_raw_signal, tokens_smoothed_signal, max_lag, significance_level
		)

		if skip_analysis:
			console.print("[red]Skipping wavelet analysis due to error with token signal.[/red]")
			continue
		# Compute signal metrics for raw and smoothed signals
		signal_metrics_df = compute_signal_metrics_for_raw_and_smoothed(
			tokens_raw_signal, tokens_smoothed_signal, merged_expanded_df, signal_data
		)
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

		volume_data = {}
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
		volume_data.update(signal_data)
		# Append to the list of volume frequencies
		volume_frequencies.append(volume_data)
		volume_df = pd.DataFrame([volume_data])
		volume_df = volume_df.drop(columns=['tokens_per_page', 'page_numbers', 'digits_per_page'])


		wavelet_results_file_path = os.path.join(wavelet_analysis_dir, f"{volume['htid'].replace('.', '_')}_wavelet_volume_results.csv")
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

	This function performs several tasks to analyze token frequency data extracted from periodicals. It processes each periodical volume, calculates token frequencies, and generates wavelet representations for raw and smoothed signals. The function also ranks wavelet configurations to determine the best representation for each volume.

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
	
	data_directory_path = get_data_directory_path()
	console.print(f"Reading preidentified periodicals from {data_directory_path}..", style="bright_blue")

	annotated_directory = os.path.join(data_directory_path, "annotated-HT-pcc-datasets", "datasets", "annotated_ht_ef_datasets")
	# Count the number of matching files
	matching_files = []
	for directory, _, files in tqdm(os.walk(annotated_directory), desc="Counting matching files"):
		for file in files:
			if file.endswith(".csv") and 'individual' in file:
				if os.path.exists(os.path.join(directory, file)):
					publication_name = directory.split("/")[-2]
					volume_number = directory.split("/")[-1]
					matching_files.append({"file": file, "directory": directory, "file_path": os.path.join(directory, file), "periodical_title": publication_name, "volume_directory": volume_number})
	matching_files_df = pd.DataFrame(matching_files)
	console.print(f"Found {len(matching_files_df)} matching files.", style="bright_green")

	volume_features_output_path = os.path.join("..", "datasets", "all_volume_features_and_frequencies.csv")
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

	# Read preidentified periodicals
	preidentified_periodicals_df = read_csv_file(os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", "periodical_metadata", "classified_preidentified_periodicals_with_full_metadata.csv"))
	periodical_titles = preidentified_periodicals_df['lowercase_periodical_name'].unique()

	# Process only annotated periodicals if specified
	for index, title in enumerate(tqdm(periodical_titles, desc="Processing periodicals")):
		console.print(f"Processing periodical {title} number ({index + 1} out of {len(periodical_titles)})..", style="magenta2")
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
			file_path = matched_row.file_path.values[0] if is_annotated_periodical else os.path.join(data_directory_path, "HathiTrust-pcc-datasets", "datasets", row.publication_directory, "volumes", row['volume_directory'], row['volume_directory'] + "_individual_tokens.csv")
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
		figures_dir_path = os.path.join("..", "figures")
		volume_frequencies = generate_signal_processing_data(volume_paths_df, output_dir=figures_dir_path, should_use_parallel=should_use_parallel, rerun_data=rerun_analysis)
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