# Wavelet Analysis for Periodicals

This project applies wavelet transformations to analyze token frequency signals in digitized periodicals. The goal is to assess segmentation quality, OCR performance, and overall signal variation across issues. This README outlines the core processing pipeline, including token extraction, stationarity testing, wavelet transformations, and ranking.

## Processing Pipeline

### 1. Token Extraction & Signal Processing

The first step is extracting token frequency signals from periodical text. This is handled in:

- `generate_signal_processing_data()`: Loads periodical data and prepares it for token extraction. The function is in the `generate_token_frequency_wavelet_analysis.py` script. It is the main entry point for processing periodicals.
- `process_tokens()`: Extracts raw and smoothed token frequency signals to capture overall trends while reducing noise. The function is in the `utils.py` script.

### 2. Stationarity Testing

Before applying wavelet transformations, we check if the token frequency signal is **stationary**—meaning its statistical properties (e.g., mean and variance) remain constant over time. Stationarity is crucial because:

- **Wavelet selection depends on stationarity**: Some wavelet methods assume stationarity, while others perform better on non-stationary signals.
- **Segmenting periodicals effectively**: Stationary signals might indicate consistent writing or typesetting, while non-stationary signals may suggest changes in style, OCR artifacts, or layout shifts.

---

#### **Stationarity Tests and Their Interpretation**

We use two complementary statistical tests to assess stationarity:

- **Augmented Dickey-Fuller (ADF) Test**:  
  - Checks if a signal has a **unit root**, which indicates non-stationarity.
  - If the **p-value is significant** (≤ `0.05`), we reject the null hypothesis, meaning the signal **is stationary**.
  - If the **p-value is not significant**, the signal is **non-stationary** and likely contains sharp changes or trends.

- **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**:  
  - Checks if a signal is **trend-stationary** (i.e., stationary after removing a deterministic trend).
  - If the **p-value is significant** (≤ `0.05`), the signal is **non-stationary**.
  - If the **p-value is not significant**, the signal **is stationary** and does not contain deterministic trends.

**Interpreting the Combined Results:**

- **ADF significant & KPSS not significant → Signal is stationary.**
- **ADF not significant & KPSS significant → Signal is non-stationary.**
- **Both significant → Potential trend-stationarity, requiring further preprocessing.**
- **Both not significant → Likely stationary, but may require additional confirmation.**

---

#### **Adaptive Testing Strategy**

Since stationarity assessments can be sensitive to parameter choices, we test across multiple configurations:

- **Iterating Over `max_lag` Values (`max_lag=[5, 10, 15]`)**:  
  - The ADF test requires a lag parameter to account for autocorrelation.
  - We evaluate the signal at **multiple lags**, ensuring that results are not sensitive to a single choice.
  - The best `max_lag` yielding stationarity is recorded.

- **Returning Detailed Interpretations**:  
  - Each test result includes an explanation of the statistical decision.
  - If different `max_lag` values produce different results, the most stationary-friendly result is used.

---

#### **Preprocessing for Stationarity**

If a signal is **non-stationary**, we apply the following transformations in sequence:

1. **Detrending**:  
   - Removes long-term trends using **linear detrending**.
   - If the signal remains non-stationary, we proceed to step 2.

2. **Constant Detrending**:  
   - Removes the mean from the signal.
   - If the signal is still non-stationary, we proceed to step 3.

3. **First-Order Differencing**:  
   - Computes the difference between consecutive values to remove trends.
   - If the signal is still non-stationary, we proceed to step 4.

4. **Second-Order Differencing**:  
   - Further removes trends by applying differencing twice.

Each transformation is tested in order, and if **any transformation produces a stationary signal**, that transformation is used for further wavelet processing.

---

#### **Implementation**

The stationarity checks and preprocessing are handled in:

- `preprocess_signal_for_stationarity()`:  
  - Iterates through **multiple stationarity transformations**.
  - Returns the **first transformation** that achieves stationarity.

- `check_wavelet_stationarity()`:  
  - Runs **ADF and KPSS tests** across multiple `max_lag` values.
  - Returns the **best configuration** that confirms stationarity.

- `apply_differencing()` & `apply_detrending()`:  
  - Apply transformations if the signal is non-stationary.

These functions are implemented in **`generate_wavelet_stationarity.py`** script.

### 3. Computing Pre-Wavelet Signal Features

After assessing stationarity, we compute various statistical and structural features of the signal. These features provide insight into the characteristics of the **raw** and **smoothed** token frequency signals, helping to guide wavelet transformation choices. The function `calculate_signal_metrics()` generates the following key features:

---

#### **Feature Table**

| Feature Name                     | Description                                                                 | Higher or Lower Better? | Interpretation for Token Frequency | Requires Stationarity? |
|----------------------------------|-----------------------------------------------------------------------------|-------------------------|-----------------------------------|------------------------|
| `avg_variance_across_levels`     | Average variance of wavelet coefficients across decomposition levels.        | Higher                  | Indicates more variation in the signal across wavelet scales. | ❌ No |
| `variance_ratio_across_levels`   | Ratio of max variance to total variance across levels.                       | Lower                   | Suggests a more evenly distributed signal across decomposition levels. | ❌ No |
| `smoothness`                     | Measures how smooth the signal is using second-order differences.            | Higher                  | Indicates a less erratic token frequency pattern. | ❌ No |
| `correlation`                     | Correlation between original and reconstructed signals.                      | Higher                  | A better preservation of the original token frequency pattern. | ✅ Yes (for reconstruction-based comparisons) |
| `dominant_frequency`             | Frequency with the highest amplitude in the FFT spectrum.                    | Context-dependent       | Higher values suggest more periodic patterns in token frequency. | ❌ No |
| `amplitude_max`                  | Maximum amplitude of the dominant frequency.                                 | Higher                  | Indicates stronger periodic patterns. | ❌ No |
| `num_fft_peaks`                  | Number of prominent peaks in the frequency spectrum.                         | Context-dependent       | More peaks suggest complex, multi-periodic structures in token frequency. | ❌ No |
| `relative_num_peaks`             | Number of peaks detected in the original token frequency signal.             | Higher                  | Suggests more fluctuation in token frequency. | ❌ No |
| `avg_prominence`                 | Average prominence of detected peaks.                                       | Higher                  | Stronger, more distinct peaks in token frequency. | ❌ No |
| `prominence_min`                 | Minimum prominence of detected peaks.                                       | Lower                   | Indicates weak, subtle variations in token frequency. | ❌ No |
| `prominence_max`                 | Maximum prominence of detected peaks.                                       | Higher                  | Indicates dominant, strong fluctuations in token frequency. | ❌ No |
| `max_autocorrelation`            | Maximum autocorrelation value of the signal.                                 | Higher                  | Suggests more self-similarity and periodicity in token frequency. | ✅ Yes (for reliable periodicity detection) |
| `upper_envelope`                 | Maximum absolute signal value.                                              | Higher                  | Indicates stronger fluctuations in token frequency. | ❌ No |
| `lower_envelope`                 | Minimum absolute signal value.                                              | Lower                   | Indicates more stability in token frequency. | ❌ No |
| `spectral_magnitude`             | Sum of all FFT amplitudes, indicating overall spectral energy.              | Higher                  | Stronger signal with more energy in frequency components. | ❌ No |
| `spectral_centroid`              | Weighted mean frequency of the signal’s spectrum.                           | Higher                  | Suggests the signal’s energy is concentrated at higher frequencies. | ❌ No |
| `spectral_bandwidth`             | Measure of how spread out the spectral energy is.                           | Higher                  | A wider spread suggests a more complex signal. | ❌ No |
| `frequency_max`                  | Highest frequency observed in the FFT spectrum.                             | Higher                  | Indicates more rapid fluctuations in token frequency. | ❌ No |

These features provide a **comprehensive view of the token frequency signal**, capturing its periodicity, variance, and structural characteristics. They serve as a foundation for evaluating wavelet transformations and identifying the most suitable representation for further analysis. This code is executed in the `generate_wavelet_signal_features.py` script.

---

#### **Key Considerations**

- **Most features do not require stationarity**, but reconstruction-based ones like `correlation` and `max_autocorrelation` are **more reliable on stationary signals**.
- **FFT-based features (`spectral_*`, `frequency_max`, etc.) are independent of wavelet transforms** and apply to both raw and smoothed signals.
- **Peak-based features (`relative_num_peaks`, `avg_prominence`, etc.) rely on a dynamic cutoff method**, ensuring meaningful peak detection across varying token frequencies.
- **All feature computations now iterate over multiple threshold values**, ensuring robust and adaptive selection.

---

#### **Feature Computation and Parameter Details**

Several parameters in `calculate_signal_metrics()` influence how these features are computed. Below is an explanation of key parameters and their roles:

##### **1. Fast Fourier Transform (FFT) Analysis**

The function `calculate_fft()` extracts spectral features from the signal, ensuring the signal is appropriate for FFT analysis.

- **Minimum Signal Length (`min_length=16`)**  
  - Ensures that the signal is long enough for FFT to produce meaningful results.

- **Signal-to-Noise Ratio (`snr_thresholds = [3.0, 5.0, 7.0]`)**  
  - Prevents performing FFT on signals dominated by noise.
  - Iterates over multiple thresholds to select the **best** SNR cutoff.

- **Stationarity Threshold (`stationarity_thresholds = [0.3, 0.5, 0.7]`)**  
  - Ensures FFT results are reliable by checking whether the power spectrum is stable.
  - Iterates over different settings to **automatically find the best threshold**.

##### **2. Peak Detection Parameters**

Peak-based metrics (`relative_num_peaks`, `avg_prominence`, etc.) are extracted using `detect_relative_peaks()`. This method applies:

- **Prominence Threshold (`prominence_values = [0.05, 0.1, 0.15]`)**  
  - Ensures only **significant** peaks are counted.
  - Iterates over **multiple values** to find the most robust setting.

- **Minimum Distance Between Peaks (`distance_factors = [10, 20, 30]`)**  
  - Prevents detecting closely spaced noise fluctuations as peaks.
  - Tests different ratios of signal length.

##### **3. Dynamic Cutoff Calculation**

The function `calculate_dynamic_cutoff()` estimates a dynamic threshold for peak detection.

- **Based on Signal Median and Peak Amplitude:**  
  - Helps differentiate between meaningful fluctuations and noise.

- **Lower Percentile Enforcement (`percentile_values = [5, 10, 15]`)**  
  - Ensures that the cutoff is not too extreme.
  - Iterates to select the best percentile dynamically.

---

#### **Implementation Details**

The signal feature computations are handled in:

- **`calculate_signal_metrics()`**  
  - Main function computing **all** the above features.
  - Uses multiple helper functions.

- **Helper Functions:**  
  - `calculate_fft()`, `analyze_fft_peaks()`: Extract spectral properties.  
  - `detect_relative_peaks()`: Identifies local peaks and their prominence.  
  - `calculate_autocorrelation()`: Measures periodicity using autocorrelation.  
  - `calculate_signal_envelope()`: Extracts upper/lower signal bounds.  
  - `calculate_spectral_features()`: Computes spectral magnitude, centroid, and bandwidth.  
  - `calculate_dynamic_cutoff()`: Determines meaningful peak detection thresholds.

These functions are implemented in **`generate_wavelet_features.py`**.

### 4. Applying Wavelet Transformations & Computing Metrics

After computing signal metrics, we evaluate the performance of different **wavelet transformations**—**Discrete Wavelet Transform (DWT), Continuous Wavelet Transform (CWT), and Stationary Wavelet Transform (SWT)**—to determine which best represents the **token frequency signal**. This evaluation is handled by `compare_and_rank_wavelet_metrics()` in the `generate_token_frequency_wavelet_analysis.py` script, which applies each wavelet transform and assesses its effectiveness using multiple criteria.

The function operates by:

- **Filtering Wavelet Types by Signal Characteristics**  
  - Since **DWT and SWT require stationarity**, they are only applied if the stationarity tests confirm a stable signal.  
  - **CWT**, which is more flexible, is applied to all signals regardless of stationarity.  
- **Applying Each Wavelet to the Signal**  
  - Each transformation is tested with **multiple parameter settings** (e.g., decomposition levels for DWT/SWT and wavelet scales for CWT).  
  - The transformed signal is reconstructed, and its fidelity is compared to the original.  
- **Computing Wavelet-Specific Metrics**
  - **Reconstruction-based metrics** (`PSNR`, `MSE`, `EMD`, `KL Divergence`) evaluate how well the wavelet preserves the original signal.
  - **Sparsity-based metrics** (`wavelet_sparsity`, `wavelet_energy_entropy`) assess signal compressibility and structure retention.

---

#### **Wavelet Evaluation Metrics**

This table documents the **key metrics** used to rank wavelet transformations, focusing on **signal reconstruction accuracy, sparsity, and information compression**.

| **Metric Name**               | **Description**                                                             | **Higher or Lower Better?** | **Applies to (CWT, DWT, SWT)** | **Requires Stationarity?** | **Preprocessing or Transformations Required** |
|--------------------------------|---------------------------------------------------------------------------|-----------------------------|---------------------------------|----------------------------|----------------------------------------------|
| `wavelet_psnr`                | Peak Signal-to-Noise Ratio (PSNR) measures how well the wavelet-reconstructed signal preserves the original. | **Higher**                  | DWT, SWT                      | ✅ Yes (for DWT/SWT)  | Requires signal reconstruction. |
| `wavelet_mse`                 | Mean Squared Error (MSE) measures the difference between original and reconstructed signals. | **Lower**                   | DWT, SWT                      | ✅ Yes (for DWT/SWT)  | Requires signal reconstruction. |
| `wavelet_energy_entropy`      | Ratio of total energy to entropy in wavelet coefficients, indicating structural complexity. | **Lower**                   | DWT, SWT, CWT                 | ❌ No                     | Computed using wavelet coefficients. |
| `wavelet_sparsity`            | Measures how many wavelet coefficients are nonzero, indicating signal compressibility. | **Higher**                  | DWT, SWT, CWT                 | ❌ No                     | Requires thresholding on wavelet coefficients. |
| `wavelet_adaptive_threshold`  | Adaptive threshold used for sparsity measurement.                            | **N/A**                     | DWT, SWT, CWT                 | ❌ No                     | Computed dynamically for each wavelet. |
| `emd_value`                   | Earth Mover's Distance (EMD) between the original and reconstructed signal.  | **Lower**                   | DWT, SWT, CWT                 | ❌ No                     | Requires reconstruction and comparison. |
| `kl_divergence`               | Kullback-Leibler Divergence between original and reconstructed signal distributions. | **Lower**                   | DWT, SWT, CWT                 | ❌ No                     | Requires positive signal values; small offsets added if needed. |

These wavelet-specific metrics provide a **quantitative basis for ranking different wavelet transformations**. Again, metrics such as `PSNR`, `MSE`, and `emd_value` directly measure reconstruction fidelity, while sparsity-based metrics (`wavelet_sparsity`, `wavelet_energy_entropy`) evaluate **signal compression and structure retention**.

---

#### **Key Considerations in Wavelet Selection**

- **Wavelet selection depends on signal characteristics**  
  - **DWT and SWT require stationarity**, while **CWT is more flexible**.  
- **Reconstruction-based metrics (`PSNR`, `MSE`, `EMD`, `KL Divergence`) require stationarity**  
  - **For DWT/SWT**, stationarity **must** be confirmed before application.  
  - **For CWT**, stationarity is **not required**, making it applicable in more cases.  
- **Entropy- and sparsity-based metrics (`wavelet_energy_entropy`, `wavelet_sparsity`) are independent of stationarity**  
  - These **help evaluate signal compressibility** without requiring reconstruction.  
- **Certain transformations require adjustments**  
  - **SWT requires padding** to enforce **even-length signals** for proper decomposition.  
  - **DWT max decomposition levels are constrained by wavelet properties**—not all wavelets work at all levels.  

---

#### **Implementation Details**

The wavelet evaluation process is handled separately for each wavelet type with an option to either run the evaluation sequentially or in parallel. The key functions are:

1. **`evaluate_dwt_performance()` & `evaluate_dwt_performance_parallel()`**  
   - Apply **DWT** to **stationary signals** at **multiple decomposition levels**.  
   - Compute **reconstruction fidelity metrics (PSNR, MSE, energy entropy, sparsity, KL divergence, etc.).**  
   - Parallel processing is available for **faster evaluation**.  

2. **`evaluate_cwt_performance()` & `evaluate_cwt_performance_parallel()`**  
   - Apply **CWT** to all signals (no stationarity requirement).  
   - Extract **wavelet coefficient-based features**, but does not perform direct reconstruction.  
   - **`scales_used` metadata** is recorded instead of needing separate scale documentation.  

3. **`evaluate_swt_performance()` & `evaluate_swt_performance_parallel()`**  
   - Apply **SWT** to **stationary signals**, ensuring **even-length signals with padding** when necessary.  
   - Compute **signal reconstruction metrics** similar to DWT.  
   - Tests **multiple decomposition levels** dynamically.  

These functions are implemented in **`generate_wavelet_transformations.py`**.

### 5. Ranking Wavelet Transformations & Metrics

The majority of this logic is in the `generate_wavelet_rankings.py` script. The script not only ranks the wavelets but also generates a JSON config file to record how all the metrics are normalized and weighted, ensuring that wavelet transformations are evaluated fairly and consistently.

#### Comparing Original vs. Reconstructed Signal Metrics

The first step in evaluating wavelet transformations is comparing the original signal metrics with the reconstructed wavelet-transformed signals. This comparison is crucial for assessing how well each wavelet transformation preserves the original signal characteristics.

Each wavelet transformation is assessed by computing differences between the original and reconstructed metrics. The comparison follows these key principles:

- **Scalar Metrics (Single Value Comparisons)**: Metrics such as amplitude, dominant frequency, and smoothness are directly compared using absolute differences. A lower difference indicates that the transformation closely preserves the original signal’s properties.
- **List-Based Metrics (Sequences and Distributions)**: Metrics like relative peaks, prominence values, and frequency distributions require more sophisticated comparisons. These are evaluated using alignment scores, Dynamic Time Warping (DTW) distance, Euclidean distance, and Wasserstein distance, depending on the metric type.
- **Handling Edge Cases**: If a reconstructed metric is missing (`NaN`), it is ignored to prevent bias. Metrics with very low variance across reconstructions are flagged and either removed or weighted lower in ranking.

By structuring the comparison this way, the analysis ensures that the best wavelet transformations are those that maintain key signal characteristics while reducing unwanted distortions. This logic is primarily handled in the `preprocess_reconstructed_signal_metrics` function.

#### Preprocessing Wavelet Signal Metrics

Once the original and reconstructed signals have been compared, the next step is preprocessing the wavelet signal metrics to ensure consistency, validity, and reliability in rankings.

A crucial part of this process is identifying and handling low-variance metrics. If a metric exhibits very little variation across wavelets, it may not meaningfully differentiate between transformations. These metrics are flagged and either penalized or ignored to prevent them from disproportionately influencing rankings. Similarly, if a metric contains negative values, such as `wavelet_energy_entropy`, a log transformation is applied to stabilize it.

Beyond adjusting individual metrics, this preprocessing step also ensures numeric consistency across the dataset. For example:

- Complex-valued metrics are converted to absolute values.
- Any extreme values or outliers are identified to prevent them from distorting rankings.

The core logic for this step is implemented in the `preprocess_wavelet_signal_metrics` function.

#### Normalization & Inversion for Fair Comparisons

After preprocessing, the final step before ranking is normalizing the metrics to ensure that comparisons are fair. Since different metrics operate on different scales, normalization ensures that no single metric dominates the rankings just because it has a larger numerical range.

Two main normalization techniques are applied:

- **Robust Scaling**, which centers values around the median and reduces the impact of outliers.
- **MinMax Scaling**, which rescales all values to fall within a range of [0,1], ensuring uniformity across metrics.

Some metrics, such as **Mean Squared Error (MSE), Wasserstein distance, and KL divergence**, are inherently better when they are lower, as they indicate less deviation from the original signal. To maintain consistency in ranking, these metrics are **inverted** so that a higher score always reflects better performance. This logic is in the `normalize_metrics` function in the `generate_wavelet_rankings.py` script.

#### Weighting Metrics for Ranking

Once all metrics are normalized, they are weighted according to their importance in ranking wavelet transformations. This weighting process occurs in multiple steps:

- **Static Weighting:** First, each metric is assigned a predefined weight based on its significance. Metrics that strongly indicate signal fidelity, such as **spectral magnitude difference and wavelet entropy**, receive higher weights, while less critical metrics, such as **relative peaks alignment scores**, are weighted lower. These are defined in the `generate_wavelet_rankings.py` script and is split between the `RECONSTRUCTION_SIGNAL_METRIC_WEIGHTS` and `WAVELET_SIGNAL_METRIC_WEIGHTS` dictionaries. The code is run from the `calculate_normalized_weighted_scores_by_metric_type` function. This code generates the first ranking columns:

```python
# Name of the columns we'll produce
weighted_col = f"{prefix}{metric_type}_weighted_score"
weighted_norm_col = f"{prefix}{metric_type}_normalized_weighted_score"
weighted_rank_col = f"{prefix}{metric_type}_normalized_weighted_rank"
```

Prefix is intended for when we are combining either across signal types within a wavelet (in which case the prefix is `across_`) or combining across all wavelets (in which case the prefix is `all_`). The `metric_type` is either `reconstruction_signal` or `wavelet_signal`, which are our groupings of metrics. The `weighted_col` is the weighted score, which is the sum of the normalized scores multiplied by their weights. The `weighted_norm_col` is the normalized weighted score, which is the weighted score divided by the sum of all weights. The `weighted_rank_col` is the rank of the normalized weighted score.

We also explore how these features correlate and diverge in the `WaveletFeatureAnalysis` notebook.

---

- **Dynamic Weight Adjustments:** Once we have our initial weighted scores, we apply dynamic adjustments based on observed data characteristics. This ensures that the final ranking reflects meaningful distinctions among wavelet transformations.

To refine the ranking process further, weights are dynamically adjusted based on observed data characteristics:

- **Metrics with High Variance and Strong Presence:**  
  If a metric appears in fewer than **10%** of the data (presence threshold) or has a variance below **1e-6** (variance threshold), it is flagged for exclusion and removed from consideration.
  
- **Metrics with Low Variance or Frequent Missing Values:**  
  These are weighted lower or penalized to prevent them from skewing rankings. The penalty factor is **0.5**, meaning that if a metric is missing too many values, it is penalized by half its weight.

- **Normalization of Weights:**  
  In addition to penalizing unreliable metrics, the dynamic weighting process ensures that **all final metric weights sum to 1**, maintaining consistency across different datasets and wavelet families. This prevents any single metric from disproportionately influencing rankings.

The function `normalize_weights_dynamically` handles this process, ensuring that the final weight distribution reflects meaningful distinctions among wavelet transformations. It also updates the ranking config file to record these adjustments.

Beyond adjusting weights dynamically, the ranking system also **tracks missing values** for each wavelet transformation and **applies a penalty**. The function `calculate_dynamically_normalized_weighted_score_by_metric_type` **counts missing values per row**. It then applies a penalty factor of **0.05** for each missing value, reducing the overall score for that wavelet transformation. This ensures that wavelet transformations with incomplete data are appropriately penalized in the final ranking.

After penalizing unreliable metrics, the dynamic weighting and scoring process normalizes the values between a range of 0 and 1. This normalization ensures that all final scores are comparable and reflect meaningful distinctions among wavelet transformations. Finally, it generates the following columns:

```python
# Name of the columns we'll produce
weighted_col = f"{prefix}{metric_type}_dynamically_weighted_score"
weighted_norm_col = f"{prefix}{metric_type}_normalized_dynamically_weighted_score"
weighted_rank_col = f"{prefix}{metric_type}_normalized_dynamically_weighted_rank"
```

The `weighted_col` is the dynamically weighted score, which is the sum of the normalized scores multiplied by their weights. The `weighted_norm_col` is the normalized dynamically weighted score, which is the dynamically weighted score divided by the sum of all weights. The `weighted_rank_col` is the rank of the normalized dynamically weighted score.
This code is run from the `calculate_dynamically_normalized_weighted_scores_by_metric_type` function.

### 6. Generating Rankings and Config Files

#### Rank Stability and Final Configurations

The final step in the wavelet analysis pipeline is generating the final rankings and config files. Before computing stability metrics, we finalize the ranking configuration by adding summary statistics for each metric. This step ensures that all computed ranks, weights, and adjustments are recorded. The ranking config is saved in a JSON file, providing a comprehensive record of the entire ranking process.

Next we select all our ranking columns that we have generated:

```python
rank_cols = [
f"{prefix}reconstruction_signal_normalized_weighted_rank", f"{prefix}reconstruction_signal_normalized_dynamically_weighted_rank", 
f"{prefix}wavelet_signal_normalized_weighted_rank", f"{prefix}wavelet_signal_normalized_dynamically_weighted_rank",
]
```

And use these to calculate the stability of ranks across these columns. The stability is calculated using the `calculate_rank_stability` function. In the function, we compute the rank variability and normalized rank stability for each wavelet transformation. The rank variability is the standard deviation of ranks across different ranking columns, while the normalized rank stability is a value between [0,1] where higher values indicate greater stability. We also compute the average and harmonic average rank across all wavelet metrics. Finally, we compute the weighted stability score, which is a combination of normalized rank stability and rank consistency. It also has a weight factor of 0.5.

The core issue with trying to compare wavelet ranks is that the `reconstruction_signal` and `wavelet_signal` metrics do not always correlate because they are measuring different things. The `reconstruction_signal` metrics are measuring how well the wavelet preserves the original signal, while the `wavelet_signal` metrics are measuring how well the wavelet captures the structure of the signal. This means that a wavelet that performs well on one set of metrics may not perform well on the other. Currently, we simply weight both equally in determining rank stability, but this is an area for future exploration though again it remains difficult without ground truth data.

Ultimately, if a wavelet transformation ranks 1st under static weighting, but 5th under dynamic weighting, the rank stability score would decrease, reflecting its inconsistent performance. Conversely, a wavelet that ranks 3rd across all ranking methods would have a higher stability score, indicating its robustness. Which is why wavelets with low rank variability are likely more reliable for representing our periodicals. The stability score is a measure of how consistent the wavelet is across different metrics. A high stability score indicates that the wavelet is consistently performing well across all metrics, while a low stability score indicates that the wavelet is performing well on some metrics but not others. This helps filter out wavelet transformations that rank highly due to noise in individual metrics and improves the robustness of the ranking pipeline by ensuring consistent selection criteria.

#### Family Rank Stability and Top Rankings

In addition to individual wavelet stability, we also assess stability at the wavelet family level (e.g., Haar, Daubechies, Coiflet). This is done by grouping wavelets by their family and computing the average rank for each family. We then compute the rank variability and normalized rank stability for each family. This helps us identify families of wavelets that are consistently performing well across all metrics. Some wavelet families, like Daubechies, are optimized for multi-resolution analysis, while others, like Coiflet, are better for smooth transitions. By ensuring at least one high-ranking wavelet per family, we improve the generalizability of selected wavelets across different periodicals.

Once we have the family stability scores, we create a `family_informed_rank` which is a combination of the `stability_rank` and the `final_family_stability_rank`:

```python
df[f"{prefix}family_informed_rank"] = (
	0.75 * df[f"{prefix}stability_rank"] 
	+ 0.25 * df[f"{prefix}final_family_stability_rank"]
)
```

We weight the stability rank more heavily because we want to prioritize wavelets that are consistently performing well across all metrics. The family stability rank is weighted less because we want to prioritize individual wavelets that are performing well, but we also want to consider the family of the wavelet. The logic of using the family is to consider how well the family of wavelets is performing overall. This helps us identify families of wavelets that are consistently performing well across all metrics. It also helps mitigate overfitting to a single high-performing wavelet that may not generalize well, and helps elucidate which families likely perform well for periodical token frequency analysis. The `calculate_wavelet_family_stability` function handles this process.

Lastly, we apply a selection process to pick the best-performing wavelets, ensuring that the top-ranked results represent both high individual stability and strong family-level performance. While we do return the full results, we also subset since we might not want to use all the results when looking across signals or wavelet families. The final selection is based on the following criteria:

- **Select Best Ranking Column**: Because we have so many potential ways to rank columns, we compute the correlation between `f"{prefix}stability_rank"` and `f"{prefix}family_informed_rank"`. If the correlation is stronger than 85%, we use the family informed rank. Otherwise, we use the stability rank. We record this in the ranking config file.
- **Top N% of Results**: Once we have our ranking column, we then select the top N% of results. This is done by first passing in the percentage of results to select (defaults to 10%) and then using that plus the length of the dataframe to determine how many results to select.
- **Best per Wavelet Family**: In addition to the top N%, we also want to ensure that at least one wavelet from each family is represented in the final selection. This helps us maintain diversity in our wavelet selection and ensures that we are not over-relying on a single family of wavelets. So we group the dataframe and ensure that any missing wavelets are added to the final selection by selecting the top wavelet from each wavelet (not just wavelet family).
- **Final Rank**: Our last step is generating a `f"{prefix}top_wavelet_rank"` column which is the rank of the final selection based on our pared down subset.

We likely need to do some more testing to see if the full vs. subset results matter much, but the general idea is that we want to at times subset the results since there can be hundreds of low performing configurations that can produce noise in the rankings. The final selection is recorded in the ranking config file.

### 7. Saving Results

This section is likely to be updated but for now this is our logic for saving these results.

#### Directory Structure & File Naming

Since we have multiple comparisions (e.g., across signal types, wavelet families, etc.), we run the ranking process multiple times. We currently save all the results at each stage in the `compare_and_rank_wavelet_metrics` function. It produces the following directory structure in the `periodical_wavelet_analysis/datasets` directory:

```bash
├── HTID_volume/
│   ├── HTID_volume_all_full_results.csv
│   ├── HTID_volume_all_full_subset.csv
│   ├── HTID_volume_all_full_ranking_config.json
│   ├── HTID_volume_all_results.csv
│   ├── HTID_volume_all_subset.csv
│   ├── HTID_volume_all_ranking_config.json
│   ├── HTID_volume_wavelet_volume_results.csv
│   ├── DWT_results/
│   │   ├── HTID_volume_across_full_ranked_results.csv
│   │   ├── HTID_volume_across_full_subset_ranked_results.csv
│   │   ├── HTID_volume_across_ranking_config.json
│   │   ├── raw_results/
│   │   │   ├── HTID_volume_full_ranked_results.csv
│   │   │   ├── HTID_volume_subset_ranked_results.csv
│   │   │   ├── HTID_volume_ranking_config.json
│   │   ├── smoothed_results/
│   │   │   ├── HTID_volume_full_ranked_results.csv
│   │   │   ├── HTID_volume_subset_ranked_results.csv
│   │   │   ├── HTID_volume_ranking_config.json
│   ├── CWT_results/
│   │   ├── (same structure as DWT_results)
│   ├── SWT_results/
│   │   ├── (same structure as DWT_results)
```

Each `HTID_volume/` directory represents a unique periodical volume. We always replace any periods with underscores in the HTID. Inside this directory, results are further divided by wavelet family (DWT, CWT, SWT) and signal type (raw, smoothed).

We also have some particular nomenclature conventions:

- **`_all_` vs. `_across_`**: The `ALL` prefix in filenames indicates that results include all signal types and wavelets, while the `ACROSS` prefix indicates that results include only across signal types but not across wavelets.
- **`_full_` vs. `_subset_`**: The `FULL` prefix indicates that results include all wavelet transformations, while the `SUBSET` prefix indicates that results include only the top N% of wavelet transformations.

Since some of our periodicals do not meet the stationarity test or might produce errors, we record all skipped results in the `raw_results/` and `smoothed_results/` directory. We only create the initial wavelet representations at this stage of within the wavelet family and signal type. We then continue to re-rank them with the `across_` or `all_` results, giving us three stages of ranks: 

1. **Within Wavelet Family and Signal Type**: This is the initial ranking of wavelets within each family and signal type. It is saved in the `raw_results/` and `smoothed_results/` directory.
2. **Across Signal Types**: This is the ranking of wavelets across all signal types within each wavelet family. It is saved in the `DWT_results/`, `CWT_results/`, and `SWT_results/` directory.
3. **Across Wavelet Families and Signal Types**: This is the final ranking of wavelets across all wavelet families and signal types. It is saved in the `HTID_volume_all_results.csv` and `HTID_volume_all_subset.csv` files.

Finally, we generate ranking_config.json files at each step to record the configuration of the ranking process. This includes the weights, normalization methods, and any adjustments made during the ranking process. This ensures that we have a complete record of how the rankings were generated and can reproduce them if needed. This logic is the `save_wavelet_results`, `process_signal_results`, and `compare_and_rank_wavelet_metrics` functions in the `generate_token_frequency_wavelet_analysis.py` script.

#### Final Output & Plots

That function is called from the `generate_signal_processing_data` function, which is the function that creates the final file: `HTID_volume_wavelet_volume_results.csv`. This file contains the following data:

- **Volume Metadata**: Information about the volume, such as the HTID; the lowercased publication name; the `table_row_index`––which is the order of which the volume was presented and scraped from the HathiTrust website; the volume classification (generated in the `HathiTrust-PCC-Datasets` repository to identify likely multi-issue volumes); and the title classification which is aggregated from the volume classification.
- **Summary Statistics**: We also include summary statistics for each volume, including `avg_tokens` and `avg_digits`, which are the average number of tokens and digits per page, respectively; `total_pages`, `total_digits`, and `total_tokens`, which are the total number of pages, digits, and tokens in the volume, respectively; and `raw_likely_covers` and `smoothed_likely_covers`, which are the number of pages detected as likely covers using raw and smoothed token frequency cutoffs, respectively. We use the `signal_metrics_df` which is our initial dataframe of signal metrics to compute these statistics (so not derived from a wavelet reconstruction). We combine the raw and smoothed results into one row in the `merged_signals` dataframe.
- **Annotated Volumes**: A subset of our data does have ground truth for the cutoff between issues, so for those volumes we plot their token frequency with the dynamic cutoffs, and save  the plots. We also check how if we use the dynamic cutoffs to segment the signal, how many annotated issues we detect versus miss with this threshold. We return the missing issues and those are saved in this file.
- **Top Wavelet Transformations**: Finally, we save the top wavelet transformations for each volume, including the wavelet type, decomposition level, and dynamic cutoffs used for segmentation. We also save the data from the stationarity tests, including the p-values and stationarity status for both the raw and smoothed signals.

This information is saved in the `HTID_volume_wavelet_volume_results.csv` file per volume. This file serves as the final summary per volume, aggregating signal characteristics, segmentation metrics, and top wavelet configurations.

We also have some additional code. After we have processed a volume, we also generate some plots and statistics across volumes. However, this code is largely out of date, so we will need to update it. The code is in the `generate_token_frequency_wavelet_analysis.py` script.


That sounds like a great approach—getting the analysis to 70-80% completeness while proactively addressing major peer review concerns before submission. Based on your wavelet logic, ranking, and saving results, I see several areas where reviewers might flag issues. Below is a breakdown of potential weaknesses and how we might address them preemptively:

⸻

🔍 Potential Peer Review Flags & Preemptive Fixes

These are ranked from most critical to minor concerns based on their likelihood of being flagged.

1. Justification of Wavelet Selection & Scope

Issue:
	•	Right now, you apply DWT, CWT, and SWT to token frequency signals, but the rationale for using these transformations over others (e.g., STFT or empirical mode decomposition) isn’t fully clear.
	•	Reviewers might ask: Why are wavelets appropriate for periodical segmentation?
	•	Why are these three transformations chosen?

Preemptive Fix:
	•	Add a short justification section on wavelet choice, explaining:
	•	Wavelets are better for non-stationary signals compared to traditional Fourier methods.
	•	You tested DWT & SWT for stationary signals and CWT for all cases due to its flexibility.
	•	Other methods (e.g., STFT) were considered but wavelets provided better temporal localization.
	•	Consider adding a brief baseline comparison to a simpler method (e.g., a moving average or FFT segmentation) to validate that wavelets improve segmentation.

⸻

2. Stationarity Preprocessing Assumptions

Issue:
	•	The stationarity testing and preprocessing steps are solid, but reviewers might challenge:
	•	The choice of preprocessing order (detrending → differencing)
	•	Whether stationarity checks are repeated after each transformation
	•	If differencing removes meaningful periodicity from token frequencies

Preemptive Fix:
	•	Clarify that stationarity is rechecked after preprocessing.
	•	Ensure that the differencing step does not destroy key periodic structures—you might check autocorrelation before and after differencing to verify that periodic patterns remain intact.

⸻

3. Wavelet Ranking & Stability Metrics

Issue:
	•	The wavelet ranking approach (static vs. dynamic weighting) is strong, but:
	•	Why is rank stability measured by standard deviation?
	•	Is rank consistency across metrics more important than raw performance?
	•	Could the stability score favor less effective but stable wavelets?

Preemptive Fix:
	•	Justify why rank variability is a useful metric—highlight that stability ensures generalizability across different datasets.
	•	Consider an additional performance-weighted stability metric—e.g., weighting stability scores by reconstruction error so that we don’t favor “stable but bad” wavelets.

⸻

4. Handling Edge Cases & Missing Data

Issue:
	•	The pipeline excludes wavelets with high missing values but does not clarify:
	•	What percentage of missing values is tolerated?
	•	If missing metrics correlate with specific wavelet families (e.g., does Coiflet systematically fail on certain signals?)

Preemptive Fix:
	•	Specify a threshold for missing data exclusion (e.g., >20% NaN → wavelet is excluded).
	•	Add a brief analysis (even if informal) to check whether certain wavelet families fail more often than others.

⸻

5. Reproducibility & Data Artifacts

Issue:
	•	The HathiTrust periodicals are a unique dataset, but:
	•	How do OCR errors affect the token frequency signal?
	•	Are some periodicals inherently noisier than others?
	•	If another researcher applied this to a different corpus, would results hold?

Preemptive Fix:
	•	Acknowledge that OCR artifacts may introduce noise in segmentation.
	•	Suggest a future direction: testing the method on another dataset (e.g., newspapers, books).

⸻

6. Computational Complexity & Parallelization

Issue:
	•	Reviewers may ask:
	•	How long does the full pipeline take per volume?
	•	How does parallelization scale for large corpora?

Preemptive Fix:
	•	Add approximate runtime estimates for:
	•	Single-threaded vs. parallel execution
	•	Expected time for a 10-volume dataset vs. 100+ volumes
	•	If feasible, include a scalability test (e.g., runtime as a function of corpus size).

⸻

7. Interpretation of Wavelet Metrics

Issue:
	•	The wavelet ranking metrics (e.g., energy entropy, KL divergence) are well-documented, but:
	•	What does a low vs. high KL divergence mean for segmentation?
	•	How do the entropy and sparsity metrics relate to segmentation accuracy?

Preemptive Fix:
	•	Add interpretation guidelines:
	•	Lower MSE, EMD, KL divergence → better reconstruction
	•	Higher energy entropy → noisier representation
	•	Higher sparsity → stronger compression

⸻

🚀 Summary: Key Fixes Before Submission

Issue	Likelihood of Review Flag	Suggested Fix
Wavelet Selection Justification	🔴 High	Add justification for wavelets vs. other methods
Stationarity Preprocessing Assumptions	🟠 Medium-High	Clarify rechecking & impact on periodicity
Wavelet Ranking & Stability Metrics	🟠 Medium-High	Justify why stability matters & check if it favors weak wavelets
Handling Edge Cases & Missing Data	🟠 Medium	Define missing data thresholds & check if failures correlate with wavelet types
Reproducibility & Data Artifacts	🟡 Medium	Acknowledge OCR noise & discuss applicability to other datasets
Computational Complexity & Parallelization	🟡 Medium-Low	Provide runtime estimates & scaling tests
Interpretation of Wavelet Metrics	🟡 Medium-Low	Add clear explanations for key ranking metrics



⸻

Final Takeaway: You’re in a Strong Position

Right now, your biggest risks are reviewers questioning wavelet choice, ranking stability, and stationarity assumptions. Addressing these with short justifications (without overcomplicating) should preempt major reviewer pushback. 🚀

This doesn’t need to be perfect before submission, but making these key clarifications now will avoid obvious rejection flags and allow reviewers to focus on constructive feedback instead of fundamental concerns.

Let me know if you want help drafting specific justifications or want to prioritize certain areas! 🎯