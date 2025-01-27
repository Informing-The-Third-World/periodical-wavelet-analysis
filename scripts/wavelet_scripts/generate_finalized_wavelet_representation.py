import os
import re
import pandas as pd
from tqdm import tqdm
import altair as alt
alt.data_transformers.disable_max_rows()
from collections import deque
import warnings
warnings.filterwarnings('ignore')
from rich.console import Console
from rich.table import Table
import numpy as np
from scipy.fft import fft
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import pywt
import matplotlib.pyplot as plt
console = Console()
import sys

sys.path.append("..")
from scripts.utils import read_csv_file, get_data_directory_path, generate_table
from scripts.wavelet_scripts.generate_token_frequency_wavelet_analysis import process_tokens


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


		subset_final_combined_volume_df

if __name__ == "__main__":
	generate_finalized_wavelets()