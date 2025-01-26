# Serial Token Frequency As Wavelet & Signal Processing Analysis Scripts Notes

This folder contains the code and documentation for analyzing token frequency data extracted from OCR text as a signal, enabling the use of signal processing techniques to extract meaningful patterns. Essentially, it treats token frequency as a one dimensional waveform, allowing us to apply signal processing techniques to extract meaningful patterns and trends.


This analysis heavily utilizes wavelet transforms and related signal processing techniques for multi-scale analysis. The majority of the code is currently in the [`generate_frequency_analysis.py` script](../segmentation_scripts/generate_token_frequency_signal_processing_analysis.py).

## PYWT & Wavelet Analysis

In the context of token frequency analysis, we use wavelet transforms to decompose the signal into its constituent wavelets. This allows us to analyze the signal at multiple scales and extract meaningful features for further analysis. We rely heavily on the `pywt` library for this analysis, which provides a range of wavelet families and modes for analysis. In fact, we use both dynamic and continuous wavelet transforms, as well as all families and signal extension modes (`zero`, `constant`, `symmetric`, `periodic`, `smooth`, `periodization`, `reflect`, `antisymmetric`, `antireflect`).

We use the multilevel discrete wavelet transform (DWT) and continuous wavelet transform (CWT) to analyze the signal at different scales and resolutions. We could use the single DWT but we use the multilevel DWT to capture more detailed information about the signal at different scales. The CWT is also useful for analyzing the signal at different scales and resolutions, but it is computationally more expensive than the DWT.

### Full List of Wavelets

| Wavelet Transform   | Family Name                        | Short Name   | Wavelets                                                                                                                                                                                                                  |
|:--------------------|:-----------------------------------|:-------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Discrete (DWT)      | Haar                               | haar         | haar                                                                                                                                                                                                                      |
| Discrete (DWT)      | Daubechies                         | db           | db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23, db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35, db36, db37, db38 |
| Discrete (DWT)      | Symlets                            | sym          | sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20                                                                                               |
| Discrete (DWT)      | Coiflets                           | coif         | coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8, coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17                                                                                             |
| Discrete (DWT)      | Biorthogonal                       | bior         | bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior2.8, bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5, bior6.8                                                                                     |
| Discrete (DWT)      | Reverse biorthogonal               | rbio         | rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6, rbio2.8, rbio3.1, rbio3.3, rbio3.5, rbio3.7, rbio3.9, rbio4.4, rbio5.5, rbio6.8                                                                                     |
| Discrete (DWT)      | Discrete Meyer (FIR Approximation) | dmey         | dmey                                                                                                                                                                                                                      |
| Continuous (CWT)    | Gaussian                           | gaus         | gaus1, gaus2, gaus3, gaus4, gaus5, gaus6, gaus7, gaus8                                                                                                                                                                    |
| Continuous (CWT)    | Mexican hat wavelet                | mexh         | mexh                                                                                                                                                                                                                      |
| Continuous (CWT)    | Morlet wavelet                     | morl         | morl                                                                                                                                                                                                                      |
| Continuous (CWT)    | Complex Gaussian wavelets          | cgau         | cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8                                                                                                                                                                    |
| Continuous (CWT)    | Shannon wavelets                   | shan         | shan                                                                                                                                                                                                                      |
| Continuous (CWT)    | Frequency B-Spline wavelets        | fbsp         | fbsp                                                                                                                                                                                                                      |
| Continuous (CWT)    | Complex Morlet wavelets            | cmor         | cmor                                                                                                                                                                                                                      |

#### CWT Families

![cwt families](https://pywavelets.readthedocs.io/en/latest/_images/plot_wavelets.png)

#### Modes

![pywt modes](https://pywavelets.readthedocs.io/en/latest/_images/plot_boundary_modes.png)

In the script, this logic is primarily handled by the following functions:

- `energy_entropy_ratio`: this function calculates the energy entropy ratio of the wavelet coefficients, and specifically how the energy of the wavelet coefficients is distributed. High values indicate concentrated energy, suggesting a strong underlying pattern, whereas low values indicate more dispersed energy, suggesting a more complex or noisy signal.
- `sparsity_measure`: this function calculates the sparsity measure of the wavelet coefficients, which quantifies how concentrated or dense the energy is in a few wavelet coefficients. A higher sparsity measure indicates a simpler signal with fewer significant components, while a lower sparsity measure indicates a more complex signal with many significant components. 

We calls these from either the `evaluate_dwt_performance` or `evaluate_cwt_performance` functions, which iterate over the wavelet families, levels, and modes to evaluate their performance on the token frequency data. In these functions, we also calculate the mean squared error (MSE) of the wavelet coefficients, which quantifies the difference between the original signal and its reconstruction from the wavelet coefficients. A lower MSE indicates a better fit between the original signal and its wavelet representation. For DWT wavelets, we calculate the MSE using numpy's `np.mean(np.square(original_signal - reconstructed_signal))` to calculate the MSE.

For CWT wavelets, we cannot calculate MSE directly, so we instead use the `energy_entropy_ratio` and `sparsity_measure` metrics to evaluate their performance. We also need to pass in `scales` for CWT wavelets. In a CWT, you analyze a signal by “scaling” a wavelet function and shifting it along the signal. The scales array defines the range of scales (frequencies) at which the wavelet transform will analyze the signal. Scales determine how the wavelet is stretched or compressed during the transformation. Smaller scales correspond to higher-frequency components of the signal (fine details). Larger scales correspond to lower-frequency components (coarser, broader patterns). We currently hardcoded the scales to be `np.arange(1, 128)` for CWT wavelets, but this can be adjusted based on the specific signal and analysis requirements.

> **Note:** Not every wavelet works for our signal, so we write any errors to a log file for later review (more on file structure below).

## Processing Tokens

In the script, we have a `process_tokens` function that gets called from the `generate_volume_embeddings` function. This function first calls `process_file` from the `utils.py` file, which reads in the initial extracted features from HathiTrust, expands the token frequencies, creates dummy page numbers since we have some volumes with gaps in page numbers, and then calculates the number of tokens per page.

> Note: **We are completely agnostic to what constitutes a token or whether certain tokens are included.** This means that we are heavily reliant on the underlying OCR processes and the quality of the OCR output.

It is this `tokens_per_page` that is our core signal for analysis. In the `process_tokens` function, we calculate both smoothed and standardized token frequencies. Smoothed tokens are calculated using a moving average with a specified window size, while standardized tokens are calculated using z-score normalization. We primarily use the raw and smoothed tokens as input data for the analysis, depending on the specific method and analysis being performed. We also ensure that there is no null or negative values in the token frequencies, as these can cause issues with the analysis.

## Comparing & Ranking Wavelets

To compare and rank wavelets, we have the `compare_and_rank_wavelet_metrics` function, which evaluates each signal type (so `raw` or `smoothed`) and then their performance across wavelet types (so `DWT` or `CWT`), using metrics such as mean squared error (MSE), sparsity measure, and energy-to-entropy ratio. These metrics help us assess how well each wavelet captures the underlying patterns in the token frequency data.

We primarily use **smoothed** or **standardized** token frequencies as input data for the analysis:

- **Smoothed Tokens** reduce small-scale fluctuations in the data, highlighting broader trends and patterns.  
- **Standardized Tokens** normalize the smoothed data for comparability and emphasize relative deviations from the mean.  

By combining these two types of token frequencies, we capture both the overall trends and deviations from those trends, providing a comprehensive view of the data.

---

## Analysis Methods

| Method                         | Use Smoothed | Use Standardized | Key Insights                                                                                     |
|--------------------------------|--------------|------------------|-------------------------------------------------------------------------------------------------|
| **Wavelet Analysis**           | ✅ (optional) | ✅               | Smoothed tokens can reduce noise, but standardized tokens highlight relative trends better for multi-scale analysis.|
| **Autocorrelation**            | ✅ (optional)  | ✅               | Smoothing may reveal clearer periodicities, but standardization is essential for comparability across signals. |
| **Peak Detection (absolute)**  | ✅           | ❌               | Absolute peaks from smoothed tokens help identify significant features like sections or transitions. |
| **Peak Detection (relative)**  | ❌           | ✅               | Standardized tokens are better for identifying deviations from trends, revealing anomalies. |
| **Spectral Centroid & Bandwidth** | ✅ (optional) | ✅               | Smoothed tokens can provide a stable signal, but standardized tokens better capture frequency variability. |
| **Signal Envelope**            | ✅           | ❌               | Smoothing emphasizes high-level trends, useful for detecting dense vs. sparse regions.  |
| **Dominant Peak Ratios**       | ✅ (optional) | ✅               | Using smoothed tokens avoids artifacts, while standardized tokens highlight relative dynamics.|
| **Dynamic Range & Energy Distribution** | ✅ (optional)  | ✅               | Smoothed tokens stabilize energy calculations; standardization emphasizes variability. |
| **Visualizing the Wave Shape** | ✅           | ❌               | Reveals the overall shape and trends in the token signal across pages.                         |
| **Clustering or Classification** | ✅         | ✅               | Combines both absolute and relative patterns for better grouping and comparisons. Both are needed: smoothed for stability, standardized for relative comparability.|

To compare the value of each method, we do normalization on the token frequencies. This allows us to compare the relative importance of each method in capturing the underlying patterns and trends in the data.

---

## Key Difference Between Smoothed and Standardized Tokens

| Feature               | Smoothed Tokens                                   | Standardized Tokens                                     |
|-----------------------|---------------------------------------------------|-------------------------------------------------------|
| **Purpose**           | Reduce small-scale fluctuations (noise).          | Normalize the smoothed data for comparability.        |
| **Transformation**    | Moving average (windowed smoothing).              | Z-score transformation (mean = 0, std = 1).           |
| **Effect**            | Smooths trends but retains the original scale.    | Centers around 0 and scales values to std-dev units.  |
| **Blanks Handling**   | Keeps blanks (0) unchanged using `.where()`.      | Blanks are standardized relative to smoothed mean/std-dev. |
| **Interpretation**    | Reflects token trends over a local window.        | Highlights deviations relative to the smoothed mean.  |
| **Values**            | Original token range but smoothed.                | Centered around 0 with no units.                      |

---

## Why Both Are Useful

### **Smoothed Tokens (`smoothed_tokens_per_page`)**
1. **Broader Layout Trends**:
   - Highlights large-scale shifts in token density, useful for detecting headers, sections, or OCR artifacts.
   - Retains original scale, making absolute counts and comparisons straightforward.
2. **Applications**:
   - Signal envelope analysis.
   - Identifying transitions, like covers or high-density content areas.
   - Input for clustering to capture absolute trends.

### **Standardized Tokens (`standardized_tokens_per_page`)**
1. **Cross-Signal Comparisons**:
   - Normalized values allow comparison of patterns across volumes or periodicals with different scales.
   - Emphasizes relative deviations (peaks/valleys) in token counts.
2. **Applications**:
   - Wavelet analysis to detect multiscale patterns.
   - Clustering or classification, especially when combined with smoothed tokens.
   - Autocorrelation and frequency-domain analyses for periodicity detection.

---

## Method Interdependence

Some methods complement each other and should be used iteratively:
- **Wavelet Analysis** (standardized) can reveal multiscale patterns, which might guide **Peak Detection** (absolute or relative).
- **Autocorrelation** results (standardized) can inform clustering by suggesting likely periodicities or issue lengths.
- **Dynamic Range and Energy Distribution** (standardized) can enhance clustering by providing additional signal characteristics.

---

## Proposed Workflow

### Wave Shape Analysis

Wave shape analysis involves examining the shape of the waveforms to identify patterns, trends, and anomalies. It is a good fit for analyzing token distributions across pages, as it can reveal underlying structures and variations in the data. Decomposing the signal into its constituent wavelets can help identify key components and extract meaningful features for further analysis, specifically it can reveal the distribution of tokens across pages.

#### Proposed Techniques

1. Run Wavelet Analysis on Individual Volumes

- Analyze each volume independently to understand how different wavelets, levels, and modes perform. There are multiple wavelet families (e.g., Daubechies, Symlets, Coiflets) and modes (e.g., 'zero', 'constant', 'periodic', 'symmetric') to explore.
- Use a dynamic approach to determine the optimal level of decomposition for each volume based on its length and the selected wavelet.
- Store metrics (MSE, sparsity, energy-to-entropy ratio) for each wavelet, level, and mode combination.

2. Rerun Analysis After Smoothing

- Account for OCR errors, blank pages, or random artifacts that could skew wavelet metrics.
- Apply a smoothing technique to the OCR frequency signal (e.g., moving average, Gaussian smoothing).
- Compare metrics before and after smoothing to see how preprocessing affects wavelet performance.
- Finally, select the best wavelet, level, and mode for each volume based on the aggregated metrics.

3. Aggregate and Compare Across Volumes in a Periodical

- Identify the most consistent wavelet parameters for each periodical.
- Aggregate metrics across all volumes in a periodical and compute summary statistics for each wavelet (e.g., mean, median, standard deviation of metrics).
- Use a consistency threshold to select the best wavelet for each periodical based on its performance across volumes.
- Store summary statistics for each wavelet (e.g., wavelet, level, mode, mean MSE, std MSE, mean sparsity).

4. Analyze Patterns Across Periodicals

- Explore how wavelet performance varies between periodicals and whether these differences can reveal structural patterns or serve as a basis for clustering/classification.
- Compare best wavelets to see if certain wavelets consistently perform better for specific types of periodicals.
- Cluster periodicals using aggregated metrics (e.g., mean MSE, sparsity, energy ratio) with techniques like K-means or hierarchical clustering.
- Visualize periodicals in a lower-dimensional space using PCA or t-SNE for classification.
- Investigate whether differences in wavelet performance correlate with known periodical attributes (e.g., layout types, content density).

##### Potential Considerations and Adjustments

1. Iterative Refinement:

- If inconsistencies arise within a periodical, inspect problematic volumes for unique issues (e.g., OCR quality, outliers) and decide whether to exclude them or adjust preprocessing.

2. Robustness Checks:

- Cross-validate wavelet metrics by splitting the signal into train/test segments to ensure consistent results.
- Use multiple metrics (e.g., MSE, sparsity, energy ratio) to evaluate wavelet performance and avoid bias from a single metric.
- Consider the trade-offs between metrics (e.g., lower MSE vs. higher sparsity) when selecting the best wavelet.
- Store results systematically to track wavelet performance across volumes and periodicals.

#### Advantages of Wave Shape Analysis

- Captures underlying structures and variations in the data, and also decomposes the signal into multiple scales allowing us to see both fine-grained and large-scale structures. This is particularly valuable for identifying patterns or anomalies in token distributions that occur at different scales. For example, high-frequency components (small scales) might highlight abrupt OCR anomalies, while low-frequency components (large scales) can capture smooth, recurring layouts across pages.
- The ability to visualize and analyze wavelet coefficients makes it easier to interpret the signal’s behavior and identify meaningful patterns. This can help us understand how token distributions vary across pages and volumes, and how these variations relate to other document attributes (e.g., layout types, OCR quality).
- Unlike purely frequency-based methods (e.g., Fourier Transform), wavelet analysis can localize and analyze variations in both time (page position) and frequency. This makes it robust to noise and artifacts in OCR signals.
- Wavelet analysis provides a variety of wavelet families (e.g., Daubechies, Symlets, Coiflets) and parameters (e.g., levels of decomposition, extension modes), allowing tailored analysis of different types of signals.

### Spectral Centroid and Bandwidth

The spectral centroid and bandwidth are useful metrics for characterizing the frequency content of a signal. They can provide insights into the distribution of energy across frequencies and the overall shape of the spectrum. In text-based OCR signals, variations in spectral centroid and bandwidth can indicate layout patterns, OCR noise, or structural features of the text such as headers, footers, or recurring elements across pages.

#### Proposed Techniques

1. Compute Spectral Centroid and Bandwidth for Each Volume

- Calculate the spectral centroid and bandwidth of the signal using the Fourier transform.
- The spectral centroid represents the center of mass of the spectrum and can indicate the dominant frequency components. A high spectral centroid indicates faster variation in token frequencies which suggests sharper transitions between layouts, while a low spectral centroid suggests a more uniform distribution and therefore a more consistent layout throughout the volume (likely text heavy periodicals).
- The spectral bandwidth measures the spread of frequencies around the centroid and can provide information about the signal's variability and complexity. A high spectral bandwidth indicates a broad range of frequencies, which could suggest a diverse token distribution across pages, while a low spectral bandwidth suggests a more focused distribution.
- Store the spectral centroid and bandwidth values for each volume, as well as a spectrogram or frequency plot for visualization.

2. Assess Whether Smoothing Affects Spectral Properties

- Apply smoothing techniques to the signal (e.g., moving average, Gaussian smoothing) and observe how the spectral centroid and bandwidth change.
- Evaluate the impact of smoothing on the frequency content and distribution of tokens across pages.
- Store the smoothed spectral centroid and bandwidth values for comparison with the original signal.

3. Compare Spectral Centroid and Bandwidth Across Volumes

- Compare the spectral centroid and bandwidth across volumes or periodicals to detect similarities or differences in token distributions.
- Use statistical measures (e.g., mean, median, standard deviation, skewness) to summarize the spectral properties of each volume or periodical. For example, we might expect that a volume with highly repetitive layouts may show narrow bandwidth and high spectral centroid, while inconsistent volumes display broader frequency spreads and lower spectral centroids.

1. Use Spectral Centroid and Bandwidth for Clustering

- Use the spectral centroid and bandwidth as features for clustering or classification tasks to group similar volumes or periodicals based on their frequency characteristics. Likely use K-means or hierarchical clustering or DBSCAN to group similar volumes together. And then PCA or t-SNE to visualize the clusters in a lower-dimensional space.
- Investigate the relationship between the spectral centroid, bandwidth, and other signal properties to understand how frequency content relates to other aspects of the data. For example, we might explore how the spectral properties correlate with OCR quality, layout types, or MARC metadata.

##### Potential Considerations and Adjustments

1. Normalization:

- Normalize the spectral centroid and bandwidth values to compare volumes or periodicals with different frequency ranges.

2. Outlier Detection:

- Identify and handle outliers in the spectral centroid and bandwidth values to ensure robust analysis.

#### Advantages of Spectral Centroid and Bandwidth Analysis

- Provides insights into the frequency content and distribution of tokens across pages, which can reveal patterns, trends, and anomalies in the data.
- Captures the overall shape of the spectrum and the dominant frequency components, allowing us to identify key characteristics of the signal.
- Offers simple yet effective features (e.g., spectral centroid, bandwidth) for clustering and classification tasks, which can help group similar volumes or periodicals based on their frequency properties.

### Signal Envelope

Computing the envelope of the signal can help visualize the amplitude variation across the domain. The Hilbert Transform can be used to extract the envelope and analyze the signal's amplitude characteristics. This can reveal patterns, trends, and anomalies in the data, such as sudden spikes or drops in token frequencies. For periodicals, the signal envelope can provide insights into the distribution of tokens across pages and help identify recurring patterns or structural elements.

#### Proposed Techniques

1. Compute the Signal Envelope

- Use the Hilbert Transform to extract the envelope of the signal, which represents the amplitude variation over time. The analytic signal combines the original signal and its Hilbert-transformed version, allowing the envelope (the amplitude variations) to be calculated. The envelope represents a smoothed version of the signal that highlights large-scale trends while filtering out small fluctuations.
- Visualize the signal envelope to identify patterns, trends, and anomalies in the data. For example, sudden spikes or drops in the envelope could indicate significant changes in token frequencies or layout patterns.
- Store the envelope values for each volume or periodical, as well as the original signal for comparison. We expect that peaks in the envelope indicate regions with significant activity or changes in token distributions. Whereas smooth sections suggest consistent token counts, while sharp transitions could reflect structural changes, OCR artifacts, or layout shifts.

2. Assess the Effect of Smoothing on the Signal Envelope

- Apply smoothing techniques to the signal (e.g., moving average, Gaussian smoothing) and observe how the envelope changes.
- Compare the smoothed envelope with the original envelope to evaluate the impact of smoothing on the amplitude characteristics of the signal.

3. Analyze and Compare the Signal Envelope Across Volumes

- Compare the signal envelope across volumes or periodicals to detect similarities or differences in the amplitude characteristics of the data.
- Use statistical measures (e.g., mean, median, standard deviation, skewness) to summarize the envelope properties of each volume or periodical. For example, we might expect that volumes with consistent layouts have smooth envelopes, while volumes with varying layouts have more irregular envelopes. We might calculate the mean envelope amplitude, the envelope variance, and the peaks per page to quantify the amplitude characteristics of each volume.
- We will likely nee to normalize the envelop amplitudes to compare volumes or periodicals with different frequency ranges, and then aggregate metrics across volumes to identify common patterns or trends. Ultimately, we hope to identify consistent patterns (or lack thereof) in envelope characteristics across volumes within a periodical.

4. Use the Signal Envelope for Clustering

- Use the signal envelope as a feature for clustering or classification tasks to group similar volumes or periodicals based on their amplitude characteristics. We might use K-means or hierarchical clustering to group similar volumes together, and then PCA or t-SNE to visualize the clusters in a lower-dimensional space.

##### Potential Considerations and Adjustments

1. Outlier Detection:

- Identify volumes with unusually high or low envelope amplitudes or variance, which may indicate structural anomalies or errors in the OCR process. Also we might need to handle noisy or missing data that could affect the envelope calculation. Hopefully smoothing will help with this.

1. Normalization:

- Normalize the envelope amplitudes to compare volumes or periodicals with different frequency ranges.

#### Advantages of Signal Envelope Analysis

- Captures large-scale trends in token distributions while filtering out fine-grained noise.
- Highlights regions of interest (e.g., peaks) that may correspond to structural changes in the document.
- Provides simple yet effective features (e.g., amplitude variance, peak count) for clustering and classification tasks.

### Autocorrelation for Periodicity

The autocorrelation function can be used to detect repeating patterns and periodicities in the data. Peaks in the autocorrelation function can indicate regularities and cycles in the token distribution. For periodicals, autocorrelation analysis can help identify recurring layouts, headers, footers, or other structural elements that repeat across pages.

#### Proposed Techniques

1. Compute the Autocorrelation Function and Identify Peaks

- Calculate the autocorrelation function of the signal to measure the similarity between the signal and its shifted versions. Peaks in the autocorrelation function indicate repeating patterns or periodicities in the data.Peaks with high amplitudes indicate strong periodicities in the data, while smaller peaks may represent weaker or less frequent patterns.
- Visualize the autocorrelation function to identify prominent peaks and assess the periodicity of the signal. Peaks at regular intervals suggest periodic layouts or recurring elements across pages.
  
2. Assess the Effect of Smoothing on the Autocorrelation Function

- Apply smoothing techniques to the signal (e.g., moving average, Gaussian smoothing) and observe how the autocorrelation function changes.

3. Analyze and Compare Autocorrelation Across Volumes

- Compare the autocorrelation function across volumes or periodicals to detect similarities or differences in the periodicity of the data.

4. Use Autocorrelation for Clustering

- Use the autocorrelation function as a feature for clustering or classification tasks to group similar volumes or periodicals based on their periodic characteristics.

### Dominant Peak Ratios

Analyzing the ratio between the dominant peak and subsequent peaks in the frequency spectrum can reveal how prominent the primary wave shape is compared to background noise or harmonics. This is useful for identifying the key frequencies driving the signal, and when it comes to pages in a periodical or volume, it can help identify the most common token distributions. 

#### Proposed Techniques

1. Compute the Dominant Peak Ratios

### Dynamic Range and Energy Distribution

The dynamic range of a signal, defined as the difference between the maximum and minimum amplitudes, can provide insights into the signal's intensity and variability. Analyzing how energy is distributed across frequencies can help identify the key components of the signal and their relative importance.

### Visualizing the Wave Shape

Visualizing the wave shape can provide an intuitive understanding of the signal's structure and dynamics. Plotting the signal in the time domain can reveal patterns and trends, while frequency domain representations like spectrograms can show the frequency content over time.




