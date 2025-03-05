# Wavelet Analysis for Periodicals

This project applies wavelet transformations to analyze token frequency signals in digitized periodicals. The goal is to assess segmentation quality, OCR performance, and overall signal variation across issues. This README outlines the core processing pipeline, including token extraction, stationarity testing, wavelet transformations, and ranking.

## Processing Pipeline

### 1. Token Extraction & Signal Processing

The first step is extracting token frequency signals from periodical text. This is handled in:

- `process_tokens()`: Extracts raw and smoothed token frequency signals to capture overall trends while reducing noise.
- `generate_token_frequency_wavelet_analysis.py`: Loads periodical data, processes token signals, and prepares them for wavelet analysis.

### 2. Stationarity Testing

Before applying wavelet transformations, we check if the token frequency signal is **stationary**—meaning its statistical properties (e.g., mean and variance) remain constant over time. Stationarity is crucial because:

- **Wavelet selection depends on stationarity**: Some wavelet methods assume stationarity, while others perform better on non-stationary signals.
- **Segmenting periodicals effectively**: Stationary signals might indicate consistent writing or typesetting, while non-stationary signals may suggest changes in style, OCR artifacts, or layout shifts.

We use two complementary tests to check for stationarity:

- **Augmented Dickey-Fuller (ADF) Test**: Checks if a signal has a unit root, which indicates non-stationarity. If the test **rejects** the null hypothesis, the signal is likely stationary.
- **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**: Tests if the signal is trend-stationary. If the test **fails to reject** the null hypothesis, the signal is likely stationary.

Both tests provide different perspectives on stationarity, and their results together help determine the best approach for further wavelet analysis. This logic is implemented in:

- `check_wavelet_stationarity()`: Runs ADF and KPSS tests and interprets the results.
- `generate_wavelet_stationarity.py`: Manages stationarity checks across multiple periodicals.

### 3. Computing Signal Metrics

After assessing stationarity, we compute various statistical and structural metrics for both the raw and smoothed signals. These metrics provide insight into the characteristics of the signal before wavelet transformation. The function `calculate_signal_metrics()` generates the following key metrics:

| Metric Name                      | Description                                                                 | Higher or Lower Better? | Interpretation for Token Frequency | Requires Stationarity? |
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

These metrics provide a comprehensive view of the token frequency signal, capturing its periodicity, variance, and structural characteristics. They serve as a basis for evaluating wavelet transformations and identifying the most suitable representation for further analysis.

#### Key Considerations

- **Most metrics do not require stationarity**, but reconstruction-based ones like `correlation` and `max_autocorrelation` are **more reliable on stationary signals**.
- **FFT-based metrics (`spectral_*`, `frequency_max`, etc.) are independent of wavelet transforms** and apply to both raw and smoothed signals.

### 4. Applying Wavelet Transformations

After computing signal metrics, we evaluate the performance of different wavelet transformations—**Discrete Wavelet Transform (DWT), Continuous Wavelet Transform (CWT), and Stationary Wavelet Transform (SWT)**—to determine which best represents the token frequency signal. This comparison is handled in `compare_and_rank_wavelet_metrics()`, which systematically applies each wavelet type to both the raw and smoothed signals and assesses their effectiveness.

The function operates by:

- **Filtering Wavelet Types by Signal Characteristics**: Since **DWT and SWT require stationarity**, they are only applied if stationarity tests confirm a stable signal. **CWT**, which is more flexible, is applied to all signals.
- **Applying Each Wavelet to the Signal**: Each transformation is evaluated based on key metrics such as reconstruction fidelity (e.g., **PSNR, energy entropy**), smoothness, and preservation of key signal structures.
- **Ranking Wavelet Results**: Using predefined ranking criteria, the function selects the top wavelet representations. This includes penalizing missing or low-variance metrics to ensure robustness.
- **Combining Across Wavelet Types**: Results from all transformations are merged, and the highest-ranked wavelet is identified as the best fit for further analysis.

By integrating this ranking process, we ensure that the selected wavelet transformation provides an **optimal balance between detail preservation and segmentation accuracy**, making it the best candidate for assessing periodical structure and OCR quality.

This table documents the key metrics computed during the wavelet ranking process, focusing on how well different wavelet transforms preserve or enhance the original token frequency signal. These metrics evaluate **reconstruction accuracy, sparsity, and information compression**, complementing the general signal metrics.

| Metric Name                   | Description                                                                 | Higher or Lower Better? | Applies to (CWT, DWT, SWT) | Requires Stationarity? | Preprocessing or Transformations Required |
|--------------------------------|-----------------------------------------------------------------------------|-------------------------|----------------------------|------------------------|------------------------------------------|
| `wavelet_psnr`                | Peak Signal-to-Noise Ratio (PSNR) measures how well the wavelet-reconstructed signal preserves the original signal. | Higher                  | DWT, SWT                     | ✅ Yes (for DWT/SWT) | Requires signal reconstruction. |
| `wavelet_mse`                 | Mean Squared Error (MSE) measures the difference between the original and reconstructed signals. | Lower                   | DWT, SWT                     | ✅ Yes (for DWT/SWT) | Requires signal reconstruction. |
| `wavelet_energy_entropy`      | Ratio of total energy to entropy in wavelet coefficients, indicating structural complexity. | Lower                   | DWT, SWT, CWT                | ❌ No                 | Computed using wavelet coefficients. |
| `wavelet_sparsity`            | Measures how many wavelet coefficients are nonzero, indicating signal compressibility. | Higher                  | DWT, SWT, CWT                | ❌ No                 | Requires thresholding on wavelet coefficients. |
| `wavelet_adaptive_threshold`  | Adaptive threshold used for sparsity measurement.                            | N/A                     | DWT, SWT, CWT                | ❌ No                 | Computed dynamically for each wavelet. |
| `emd_value`                   | Earth Mover's Distance (EMD) between the original and reconstructed signal.  | Lower                   | DWT, SWT, CWT                | ❌ No                 | Requires reconstruction and comparison. |
| `kl_divergence`               | Kullback-Leibler Divergence between the original and reconstructed signal distributions. | Lower                   | DWT, SWT, CWT                | ❌ No                 | Requires positive signal values; small offsets added if needed. |

These wavelet-specific metrics provide a **quantitative basis for ranking different wavelet transformations**. Metrics such as `PSNR`, `MSE`, and `correlation` directly measure fidelity, while sparsity-based metrics evaluate signal compression and energy preservation.

#### Key Considerations

- **Wavelet selection depends on signal characteristics**: DWT and SWT are suitable for stationary signals, while CWT is more flexible.
- **Reconstruction-based metrics (`PSNR`, `MSE`, `EMD`, `KL Divergence`) require stationarity** for DWT/SWT but are flexible for CWT.
- **Entropy- and sparsity-based metrics** (such as `wavelet_energy_entropy` and `wavelet_sparsity`) do not require stationarity.
- **Some transformations require padding or adjustments** (e.g., SWT enforces even-length signals).


### 5. Comparing and Ranking Wavelet Transformations

The majority of this logic is in the `generate_wavelet_rankings.py` script. The script not only ranks the wavelets, but also generates a json config file to record how all the metrics are normalized and weighted to ensure that we can compare the wavelets in a fair and consistent manner.

#### Comparing Original vs. Reconstructed Metrics

The first step in evaluating wavelet transformations is comparing the original signal metrics with the reconstructed wavelet-transformed signals. This ensures that the transformations preserve key characteristics while improving the desired properties of the signal.

Each wavelet transformation is assessed by computing differences between the original and reconstructed metrics. The comparison follows these key principles:

- **Scalar Metrics (Single Value Comparisons)**: Metrics such as amplitude, dominant frequency, and smoothness are directly compared using absolute differences. A lower difference indicates that the transformation closely preserves the original signal’s properties.
- **List-Based Metrics (Sequences and Distributions)**: Metrics like relative peaks, prominence values, and frequency distributions require more sophisticated comparisons. These are evaluated using alignment scores, Dynamic Time Warping (DTW) distance, Euclidean distance, and Wasserstein distance, depending on the metric type.
- **Handling Edge Cases**: If a reconstructed metric is missing (NaN), it is ignored to prevent bias. Metrics with very low variance across reconstructions are flagged and either removed or weighted lower in ranking.

#### Normalization for Fair Comparisons

Since different metrics operate on different scales, they are normalized to ensure fair ranking:

- **Handling Low-Variance Metrics**: Metrics with near-zero variance across wavelets are either penalized or ignored to prevent skewed rankings.
- **Log Transformations for Stability**: If a metric contains negative values (e.g., wavelet_energy_entropy), a log transformation is applied to stabilize it.
- **Scaling for Uniformity**: Two normalization techniques are applied:
  - Robust Scaling: Reduces outlier effects by centering around the median.
  - MinMax Scaling: Ensures all metrics are mapped to a [0,1] range.
- Then metrics where lower values indicate better preservation (e.g., MSE, Wasserstein distance) are inverted so that higher scores always reflect better performance.

By structuring the comparison this way, the analysis ensures that the best wavelet transformations are those that maintain key signal characteristics while reducing unwanted distortions. This logic is primarily in the `normalize_metrics` function.

