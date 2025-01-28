import altair as alt
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import sys
# Local application imports
sys.path.append("../..")
from scripts.utils import generate_table, check_if_actual_issue

# Disable max rows for Altair
alt.data_transformers.disable_max_rows()

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