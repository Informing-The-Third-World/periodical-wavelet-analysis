import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import umap
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
from rich.console import Console

console = Console()

def evaluate_clusters(pivot_df, max_k=10, n_runs=5):
	"""
	Evaluates cluster stability across multiple runs.

	Parameters:
	-----------
	pivot_df : DataFrame
		The data to cluster.
	max_k : int
		Maximum number of clusters to try.
	n_runs : int
		Number of times to repeat clustering with different seeds.

	Returns:
	--------
	dict
		Dictionary containing silhouette scores, elbow values, and the most stable number of clusters.
	"""

	all_silhouette_scores = []
	all_elbow_dfs = []
	optimal_k_values = []

	for _ in range(n_runs):
		silhouette_scores = []
		elbow_df = pd.DataFrame(columns=['k', 'inertia'])

		for k in range(2, max_k + 1):
			if k >= len(pivot_df):
				continue  # Skip if the number of clusters is greater than or equal to the number of samples
			kmeans = KMeans(n_clusters=k, random_state=np.random.randint(1000))
			cluster_labels = kmeans.fit_predict(pivot_df)

			# Compute silhouette score
			if len(set(cluster_labels)) > 1:  # Ensure there is more than one cluster
				

				silhouette_avg = silhouette_score(pivot_df, cluster_labels)
				silhouette_scores.append(silhouette_avg)
			else:
				silhouette_scores.append(np.nan)
			# Compute inertia for elbow method
			new_row = pd.DataFrame({'k': [k], 'inertia': [kmeans.inertia_]})
			elbow_df = pd.concat([elbow_df, new_row], ignore_index=True)

		elbow_df['k'] = elbow_df['k'].astype(int)
		elbow_df['inertia'] = elbow_df['inertia'].astype(int)

		# Detect elbow point
		kn = KneeLocator(elbow_df['k'], elbow_df['inertia'], curve='convex', direction='decreasing')
		optimal_k = kn.knee if kn.knee else max_k

		optimal_k_values.append(optimal_k)
		all_silhouette_scores.append(silhouette_scores)
		all_elbow_dfs.append(elbow_df)

	# Find the most frequently occurring optimal cluster number
	most_common_k = Counter(optimal_k_values).most_common(1)[0][0]

	return {
		'silhouette_scores': all_silhouette_scores,
		'elbow_dfs': all_elbow_dfs,
		'stable_optimal_k': most_common_k
	}


def plot_silhouette_scores(silhouette_scores_list, max_k):
	"""
	Plot the mean silhouette scores across multiple runs.
	
	:param silhouette_scores_list: List of silhouette score lists from multiple runs.
	:param max_k: Maximum number of clusters.
	"""
	avg_silhouette_scores = np.mean(silhouette_scores_list, axis=0)
	std_silhouette_scores = np.std(silhouette_scores_list, axis=0)

	plt.figure(figsize=(8, 5))
	plt.plot(range(2, max_k + 1), avg_silhouette_scores, marker='o', label="Mean Silhouette Score")
	plt.fill_between(
		range(2, max_k + 1),
		avg_silhouette_scores - std_silhouette_scores,
		avg_silhouette_scores + std_silhouette_scores,
		color='gray', alpha=0.2, label="Std Dev"
	)
	
	plt.xlabel('Number of clusters')
	plt.ylabel('Silhouette score')
	plt.title('Mean Silhouette Scores Across Runs')
	plt.grid(True)
	plt.xticks(range(2, max_k + 1))
	plt.legend()
	plt.show()

def plot_elbow_curve(elbow_dfs):
	"""
	Plot elbow curves from multiple runs to visualize variability.
	
	:param elbow_dfs: List of DataFrames containing 'k' and 'inertia' values for different runs.
	"""
	plt.figure(figsize=(8, 5))

	# Plot all runs
	for i, df in enumerate(elbow_dfs):
		plt.plot(df['k'], df['inertia'], marker='o', alpha=0.3, label=f"Run {i+1}")

	# Compute mean elbow curve
	avg_inertia = np.mean([df['inertia'].values for df in elbow_dfs], axis=0)
	plt.plot(elbow_dfs[0]['k'], avg_inertia, marker='o', color='black', linewidth=2, label="Mean Inertia")

	plt.xlabel('Number of clusters')
	plt.ylabel('Inertia')
	plt.title('Elbow Method Across Runs')
	plt.legend()
	plt.grid(True)
	plt.show()

def create_clusters(
	pivot_df, comparison_title, type_clustering, clustering_method="KMeans", max_k=10, n_runs=5, hdbscan_params=None, show_plots=False
):
	"""
	Create and visualize clusters using KMeans, DBSCAN, or HDBSCAN and determine the most stable number of clusters.

	Parameters:
	-----------
	pivot_df : DataFrame
		The data to cluster.
	comparison_title : str
		Title for comparison.
	type_clustering : str
		Clustering type ('UMAP' or 'PCA').
	clustering_method : str
		Clustering method to use ('KMeans', 'DBSCAN', or 'HDBSCAN').
	max_k : int
		Maximum number of clusters to try (only for KMeans).
	n_runs : int
		Number of times to repeat clustering with different seeds (only for KMeans).
	hdbscan_params : dict
		Parameters for HDBSCAN clustering (e.g., {'min_cluster_size': 5, 'min_samples': None}).
	show_plots : bool
		Whether to show plots.

	Returns:
	--------
	DataFrame with cluster assignments.
	"""
	if clustering_method == "KMeans":
		# Evaluate clusters and determine the most stable number of clusters
		cluster_results = evaluate_clusters(pivot_df, max_k=max_k, n_runs=n_runs)
		stable_k = cluster_results['stable_optimal_k']
		console.print(f"Most stable number of clusters: {stable_k}", style="bright_green")

		if show_plots:
			# Plot silhouette scores and elbow curve
			plot_silhouette_scores(cluster_results['silhouette_scores'], max_k)
			plot_elbow_curve(cluster_results['elbow_dfs'])

		# Perform final clustering
		kmeans = KMeans(n_clusters=stable_k, random_state=42)
		clusters = kmeans.fit_predict(pivot_df)

	elif clustering_method == "DBSCAN":
		# Perform DBSCAN clustering
		dbscan_params = dbscan_params or {'eps': 0.5, 'min_samples': 5}
		console.print(f"Using DBSCAN with params: {dbscan_params}", style="bright_cyan")
		dbscan = DBSCAN(**dbscan_params)
		clusters = dbscan.fit_predict(pivot_df)

	elif clustering_method == "HDBSCAN":
		# Perform HDBSCAN clustering
		hdbscan_params = hdbscan_params or {'min_cluster_size': 2, 'min_samples': None}
		console.print(f"Using HDBSCAN with params: {hdbscan_params}", style="bright_cyan")
		hdbscan_clusterer = hdbscan.HDBSCAN(**hdbscan_params)
		clusters = hdbscan_clusterer.fit_predict(pivot_df)

	else:
		raise ValueError("Invalid clustering method. Choose 'KMeans', 'DBSCAN', or 'HDBSCAN'.")

	# Add cluster assignments to the pivot dataframe
	col_name = f'{clustering_method.lower()}_{type_clustering.lower()}_cluster'
	pivot_df[col_name] = clusters
	# Use UMAP to reduce dimensionality for visualization
	if type_clustering == "UMAP" and show_plots:
		reducer = umap.UMAP(random_state=42)
		if col_name in pivot_df.columns:
			cluster_pivot_df = pivot_df.drop(col_name, axis=1)
		else:
			cluster_pivot_df = pivot_df
		embedding = reducer.fit_transform(cluster_pivot_df)

		# Create a DataFrame for the embedding
		embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
		embedding_df['cluster'] = clusters

		# Visualize the clusters
		sns.scatterplot(data=embedding_df, x='UMAP1', y='UMAP2', hue='cluster', palette='viridis')


	elif type_clustering == "PCA" and show_plots:
		# Reduce the dimensionality for visualization
		# Determine the number of components for 95% variance
		pca = PCA()
		if col_name in pivot_df.columns:
			cluster_pivot_df = pivot_df.drop(col_name, axis=1)
		else:
			cluster_pivot_df = pivot_df
		pca.fit(cluster_pivot_df)
		cumsum = np.cumsum(pca.explained_variance_ratio_)
		d = np.argmax(cumsum >= 0.95) + 1
		if d < 2:
			d = 2  # Ensure at least two components are retained
		if d > 3:
			d = 3
		pca = PCA(n_components=d)
		reduced_data = pca.fit_transform(cluster_pivot_df)

		# Create a DataFrame for the reduced data
		reduced_df = pd.DataFrame(reduced_data, columns=['PCA' + str(i) for i in range(1, d+1)])
		reduced_df['cluster'] = clusters

		# Debugging: Print the columns of reduced_df
		console.print("Columns in reduced_df:", reduced_df.columns, style="bright_magenta")

		# Visualize the clusters
		if d == 3:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')

			ax.scatter(reduced_df['PCA1'], reduced_df['PCA2'], reduced_df['PCA3'], c=reduced_df['cluster'])

			ax.set_xlabel('PCA1')
			ax.set_ylabel('PCA2')
			ax.set_zlabel('PCA3')
		else:
			sns.scatterplot(data=reduced_df, x='PCA1', y='PCA2', hue='cluster', palette='viridis')
	if show_plots:
		plt.title(f'Clusters of {comparison_title} with {type_clustering}')
		plt.show()

	return pivot_df

def compute_correlation(df, norm_cols, use_thresholding=False, fill_na=True, threshold=0.1, drop_na=False):
	"""
	Computes the correlation matrix for selected numeric columns and 
	drops columns/rows if more than (1 - threshold)% of values are NaN.

	Parameters:
	-----------
	df (pd.DataFrame): Input DataFrame.
	norm_cols (list): List of column names to include in the correlation matrix.
	use_thresholding (bool): Whether to drop columns/rows based on NaN thresholding.
	threshold (float): Minimum proportion of non-NaN values required to keep a column/row.
	fill_na (bool): If True, fills remaining NaNs with 0.
	drop_na (bool): If True, drops all remaining NaNs.

	Returns:
	--------
	pd.DataFrame: Cleaned correlation matrix.
	"""

	# Step 1: Compute correlation matrix
	numeric_cols = df[norm_cols].select_dtypes(include="number").columns.tolist()
	correlation_df = df[numeric_cols].corr()

	if use_thresholding:
		# Step 2: Identify columns/rows exceeding NaN threshold
		dropped_cols = correlation_df.columns[correlation_df.isna().sum() > len(correlation_df) * (1 - threshold)].tolist()
		dropped_rows = correlation_df.index[correlation_df.isna().sum(axis=1) > len(correlation_df.columns) * (1 - threshold)].tolist()

		if dropped_cols:
			console.print(f"🛑 Dropping {len(dropped_cols)} columns due to NaNs before transposing: {dropped_cols}", style="red3")
		if dropped_rows:
			console.print(f"🛑 Dropping {len(dropped_rows)} rows due to NaNs before transposing: {dropped_rows}", style="red3")

		correlation_df.drop(columns=dropped_cols, errors='ignore', inplace=True)
		correlation_df.drop(index=dropped_rows, errors='ignore', inplace=True)

	# Step 3: Transpose matrix
	correlation_df = correlation_df.T

	# Step 4: Post-Transpose NaN Handling
	remaining_na_cols = correlation_df.columns[correlation_df.isna().sum() > 0].tolist()
	remaining_na_rows = correlation_df.index[correlation_df.isna().sum(axis=1) > 0].tolist()

	if remaining_na_cols or remaining_na_rows:
		console.print(f"⚠️ Warning: Some NaNs still present after transposing!", style="yellow")
		if remaining_na_cols:
			console.print(f"⚠️ Columns with NaNs after transposing: {remaining_na_cols}", style="yellow")
		if remaining_na_rows:
			console.print(f"⚠️ Rows with NaNs after transposing: {remaining_na_rows}", style="yellow")

		# Drop remaining NaNs if either `drop_na=True` OR `use_thresholding=True`
		if drop_na or use_thresholding:
			correlation_df.drop(columns=remaining_na_cols, errors='ignore', inplace=True)
			correlation_df.drop(index=remaining_na_rows, errors='ignore', inplace=True)
			console.print("✅ Remaining NaNs dropped after transposing!", style="bright_green")

	# Step 5: Fill NaNs if requested
	if fill_na:
		correlation_df.fillna(0, inplace=True)  # Fill remaining NaNs with 0 if requested

	return correlation_df

def run_correlations_clustering(df, norm_reconstruction_cols, data_type, use_thresholding, fill_na):
	norm_corr_df = compute_correlation(df, norm_reconstruction_cols, use_thresholding=use_thresholding, fill_na=fill_na)
	norm_pivoted_df = create_clusters(norm_corr_df, f"Reconstruction Configurations {data_type}", "PCA")
	norm_pivoted_df = create_clusters(norm_pivoted_df, f"Reconstruction Configurations {data_type}", "UMAP", max_k=15)
	norm_pivoted_df = create_clusters(norm_pivoted_df, f"Reconstruction Configurations {data_type}", "PCA", "HDBSCAN", max_k=15)
	norm_pivoted_df = create_clusters(norm_pivoted_df, f"Reconstruction Configurations {data_type}", "UMAP", "HDBSCAN", max_k=15)
	norm_pivoted_df['metric'] = norm_pivoted_df.index
	norm_pivoted_df = norm_pivoted_df.reset_index(drop=True)
	return norm_pivoted_df

def combined_across_clusters(df, cluster_cols):
	finalized_df = []
	initial_cluster = cluster_cols[0]
	clusters = df[initial_cluster].unique()

	for idx, cluster in enumerate(clusters):
		final_clustered_cols = []
		initial_metrics = df[(df[initial_cluster] == cluster)][[initial_cluster, 'metric']]
		initial_metrics['cluster_type'] = initial_cluster
		initial_metrics = initial_metrics.rename(columns={initial_cluster: 'original_cluster'})
		final_clustered_cols.append(initial_metrics)
		for col in cluster_cols[1:]:
			comparison_clusters = df[(df[col] == cluster)][col].unique().tolist()
			if len(comparison_clusters) > 2:
				console.print(f"Cluster {cluster} has more than 2 clusters in {col}. Should be checked manually", style="red")
			comparison_metrics = df[(df['kmeans_pca_cluster'].isin(comparison_clusters))][['metric', col]]
			comparison_metrics['cluster_type'] = col
			comparison_metrics = comparison_metrics.rename(columns={col: 'original_cluster'})
			final_clustered_cols.append(comparison_metrics)
		# concat dataframe and drop duplicates
		final_clustered_cols = pd.concat(final_clustered_cols)
		final_clustered_cols = final_clustered_cols.drop_duplicates()
		final_clustered_cols['new_cluster'] = idx
		finalized_df.append(final_clustered_cols)
	return pd.concat(finalized_df)
	
def compute_correlation_scores(df, norm_cols, data_type, signal_type):
	numeric_cols = df[norm_cols].select_dtypes(include="number").columns.tolist()
	correlation_df = df[numeric_cols].corr()
	subset_correlation_df = correlation_df.loc[['reconstruction_score_sum', 'wavelet_summed_norm_score', "reconstruction_score_weighted", 'final_score', 'summed_scores']]
	# subset_correlation_df = subset_correlation_df.dropna(axis=1, how='all')

	subset_correlation_df = subset_correlation_df.T.sort_values(by='summed_scores', ascending=False)
	subset_correlation_df['metric'] = subset_correlation_df.index
	subset_correlation_df = subset_correlation_df.reset_index(drop=True)
	subset_correlation_df['data_type'] = data_type
	subset_correlation_df['signal_type'] = signal_type
	subset_correlation_df = subset_correlation_df[['metric', 'reconstruction_score_sum', 'wavelet_summed_norm_score',
       "reconstruction_score_weighted", 'final_score', 'summed_scores', 'data_type', 'signal_type']]
	return subset_correlation_df

def test_stability_clusters(grouped_full_raw_df, grouped_subset_raw_df, grouped_full_smoothed_df, grouped_subset_smoothed_df):
	
	# Dictionary to store results
	stable_groups = []
	unstable_groups = []

	# Create a copy of unique_metrics to modify it in-place
	unique_metrics = grouped_full_raw_df.metric.unique().tolist()

	# Loop through each metric and compare its groupings
	for metric in unique_metrics[:]:  # Iterate over a copy to allow removal
		# Get the cluster number for this metric in each data type
		initial_full_raw_cluster = grouped_full_raw_df[grouped_full_raw_df.metric == metric].new_cluster.unique()[0]
		initial_subset_raw_cluster = grouped_subset_raw_df[grouped_subset_raw_df.metric == metric].new_cluster.unique()[0]
		initial_full_smoothed_cluster = grouped_full_smoothed_df[grouped_full_smoothed_df.metric == metric].new_cluster.unique()[0]
		initial_subset_smoothed_cluster = grouped_subset_smoothed_df[grouped_subset_smoothed_df.metric == metric].new_cluster.unique()[0]

		# Get the set of metrics in each cluster
		full_raw_metrics = set(grouped_full_raw_df[grouped_full_raw_df.new_cluster == initial_full_raw_cluster].metric.tolist())
		subset_raw_metrics = set(grouped_subset_raw_df[grouped_subset_raw_df.new_cluster == initial_subset_raw_cluster].metric.tolist())
		full_smoothed_metrics = set(grouped_full_smoothed_df[grouped_full_smoothed_df.new_cluster == initial_full_smoothed_cluster].metric.tolist())
		subset_smoothed_metrics = set(grouped_subset_smoothed_df[grouped_subset_smoothed_df.new_cluster == initial_subset_smoothed_cluster].metric.tolist())

		# Compare all sets to check if they are identical
		all_sets = [full_raw_metrics, subset_raw_metrics, full_smoothed_metrics, subset_smoothed_metrics]

		# Check if all sets are identical by comparing them pairwise
		sets_are_equal = all(s == all_sets[0] for s in all_sets)

		# Handle stable vs unstable cases
		if sets_are_equal:
			unique_metrics.remove(metric)  # Remove from future checks
			stable_groups.append({
				'cluster_raw': list(full_raw_metrics), 
				'cluster_smoothed': list(full_smoothed_metrics),
				'metric': metric, 
				'cluster_number_raw': initial_full_raw_cluster,
				'cluster_number_smoothed': initial_full_smoothed_cluster
			})
		else:
			unstable_groups.append({
				'metric': metric,
				'full_raw_vs_subset_raw': list(full_raw_metrics.symmetric_difference(subset_raw_metrics)),
				'full_smoothed_vs_subset_smoothed': list(full_smoothed_metrics.symmetric_difference(subset_smoothed_metrics)),
				'full_raw_vs_full_smoothed': list(full_raw_metrics.symmetric_difference(full_smoothed_metrics)),
				'subset_raw_vs_subset_smoothed': list(subset_raw_metrics.symmetric_difference(subset_smoothed_metrics)),
				'cluster_raw': list(full_raw_metrics),
				'cluster_smoothed': list(full_smoothed_metrics),
				'cluster_number_raw': initial_full_raw_cluster,
				'cluster_number_smoothed': initial_full_smoothed_cluster
			})

	# Convert unstable_groups to DataFrame for easier analysis
	unstable_df = pd.DataFrame(unstable_groups) if len(unstable_groups) > 0 else pd.DataFrame()
	stable_df = pd.DataFrame(stable_groups) if len(stable_groups) > 0 else pd.DataFrame()
	return unstable_df, stable_df


