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

### 3. Computing Signal Features

After assessing stationarity, we compute various statistical and structural features of the signal. These features provide insight into the characteristics of the **raw** and **smoothed** token frequency signals, helping to guide wavelet transformation choices. The function `calculate_signal_metrics()` generates the following key features:

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

These features provide a comprehensive view of the **token frequency signal**, capturing its periodicity, variance, and structural characteristics. They serve as a foundation for evaluating wavelet transformations and identifying the most suitable representation for further analysis. This code is executed in the `generate_wavelet_features.py` script.

#### **Key Considerations**

- **Most features do not require stationarity**, but reconstruction-based ones like `correlation` and `max_autocorrelation` are **more reliable on stationary signals**.
- **FFT-based features (`spectral_*`, `frequency_max`, etc.) are independent of wavelet transforms** and apply to both raw and smoothed signals.
- **Peak-based features (`relative_num_peaks`, `avg_prominence`, etc.) rely on a dynamic cutoff method**, ensuring meaningful peak detection across varying token frequencies.

#### **Feature Computation and Parameter Details**

Several parameters in `calculate_signal_metrics()` influence how these features are computed. Below is an explanation of key parameters and their roles:

##### **1. Fast Fourier Transform (FFT) Analysis**

The function `calculate_fft()` is used to extract spectral features from the signal.

- **Minimum Signal Length (`min_length=16`)**  
  - Ensures that the signal is long enough for FFT to produce meaningful results.

- **Signal-to-Noise Ratio (`snr_threshold=5.0 dB`)**  
  - Prevents performing FFT on signals dominated by noise.

- **Stationarity Threshold (`stationarity_threshold=0.5`)**  
  - Ensures that FFT results are reliable by checking whether the power spectrum is stable.

##### **2. Peak Detection Parameters**

Peak-based metrics (`relative_num_peaks`, `avg_prominence`, etc.) are extracted using `detect_relative_peaks()`. This method applies:

- **Prominence Threshold (`prominence=10% of signal’s standard deviation`)**  
  - Ensures only significant peaks are counted.

- **Minimum Distance Between Peaks (`distance = signal length / 20`)**  
  - Prevents detecting closely spaced noise fluctuations as peaks.

##### **3. Dynamic Cutoff Calculation**

The function `calculate_dynamic_cutoff()` estimates a dynamic threshold for peak detection.

- **Based on Signal Median and Peak Amplitude:**  
  - Helps differentiate between meaningful fluctuations and noise.

- **Lower Percentile Enforcement (`10th percentile of the signal`)**  
  - Ensures that the cutoff is not too extreme.

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

