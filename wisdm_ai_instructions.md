# WISDM-51 Activity Recognition - Complete AI Agent Instructions

## Project Overview
You are working on a Human Activity Recognition (HAR) project using the WISDM-51 dataset. The project involves processing sensor data from smartphones and smartwatches to classify 18 different human activities.

## Current Status Summary

### ✅ COMPLETED (Weeks 1-5)
- **Step 1: Data Cleaning** - Raw data loaded and cleaned
- **Step 2: Windowing** - 3-second windows (60 samples @ 20Hz) with 50% overlap
- **Step 3: Time-Domain Feature Extraction** - 60 features extracted (20 per axis: x, y, z)
- **Step 4: Scaling** - MinMax, Standard, and Robust scaling applied
- **Step 5: Feature Selection** - Variance threshold + Mutual Information (top 30 features)
- **Step 6: Model Training** - KNN, Naive Bayes, Decision Tree, Random Forest trained
- **Step 7: Results Summary** - Best model: MinMax + RandomForest (72.21% accuracy)

### 🔲 REMAINING WORK (Weeks 6-9)

You need to implement **PART 3** which focuses on **Spectral (Frequency-Domain) Features**.

---

## TASK BREAKDOWN - WHAT YOU NEED TO DO

### **Task 1: Compute Spectral Features**

Create a new file: `step8_spectral_features.py`

**Requirements:**
1. Load the windowed data from `data/02_windowed/windowed_data.csv`
2. For each window and each axis (x, y, z), compute the following spectral features using FFT:

#### Spectral Descriptors (7 features per axis = 21 total)
- **Spectral Energy**: Sum of squared magnitudes in frequency domain
- **Spectral Entropy**: Measure of spectral complexity/randomness
- **Spectral Centroid**: Center of mass of the spectrum (weighted mean frequency)
- **Spectral Spread**: Standard deviation around the spectral centroid
- **Spectral Flux**: Rate of change between consecutive spectra
- **Spectral Roll-off**: Frequency below which 85% of spectral energy lies
- **Spectral Flatness**: Ratio of geometric mean to arithmetic mean (tonality vs noise)

#### Peak-Related Features (6 features per axis = 18 total)
- **Dominant Frequency**: Frequency with highest magnitude
- **Frequency of Top-3 Peaks**: Three frequencies with largest peaks
- **Amplitude of Dominant Frequency**: Magnitude at dominant frequency
- **Bandpower 0-5 Hz**: Total power in 0-5 Hz range
- **Bandpower 5-10 Hz**: Total power in 5-10 Hz range
- **Periodicity**: Measure of signal repetitiveness
- **Fundamental Frequency**: Lowest frequency component
- **Harmonic Ratio**: Ratio of harmonic to non-harmonic components

**Total**: 39 features per axis × 3 axes = **117 spectral features**

**Implementation Guidelines:**
```python
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

def compute_spectral_features(signal, sampling_rate=20):
    """
    Compute all spectral features for a single axis
    
    Parameters:
    -----------
    signal : array-like
        Time-domain signal (60 samples)
    sampling_rate : int
        Sampling rate in Hz (default: 20)
    
    Returns:
    --------
    dict : Dictionary of spectral features
    """
    features = {}
    
    # Compute FFT
    N = len(signal)
    fft_vals = fft(signal)
    fft_freqs = fftfreq(N, 1/sampling_rate)
    
    # Use only positive frequencies
    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    magnitudes = np.abs(fft_vals[pos_mask])
    
    # Normalize magnitudes for probability distribution
    mag_sum = np.sum(magnitudes)
    if mag_sum > 0:
        psd = magnitudes / mag_sum
    else:
        psd = magnitudes
    
    # 1. Spectral Energy
    features['spectral_energy'] = np.sum(magnitudes ** 2)
    
    # 2. Spectral Entropy
    psd_nonzero = psd[psd > 0]
    features['spectral_entropy'] = -np.sum(psd_nonzero * np.log2(psd_nonzero)) if len(psd_nonzero) > 0 else 0
    
    # 3. Spectral Centroid
    features['spectral_centroid'] = np.sum(freqs * psd) if mag_sum > 0 else 0
    
    # 4. Spectral Spread
    centroid = features['spectral_centroid']
    features['spectral_spread'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd)) if mag_sum > 0 else 0
    
    # 5. Spectral Flux (use previous spectrum or zeros)
    # Note: For first window, compare with zeros
    features['spectral_flux'] = np.sum((magnitudes - 0) ** 2) ** 0.5
    
    # 6. Spectral Roll-off (85% threshold)
    cumsum = np.cumsum(magnitudes)
    rolloff_threshold = 0.85 * cumsum[-1]
    rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
    features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    # 7. Spectral Flatness
    geometric_mean = np.exp(np.mean(np.log(magnitudes + 1e-10)))
    arithmetic_mean = np.mean(magnitudes)
    features['spectral_flatness'] = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
    
    # Peak-related features
    peaks, properties = find_peaks(magnitudes, height=np.mean(magnitudes))
    
    # 8. Dominant Frequency
    if len(magnitudes) > 0:
        dominant_idx = np.argmax(magnitudes)
        features['dominant_frequency'] = freqs[dominant_idx]
        features['dominant_amplitude'] = magnitudes[dominant_idx]
    else:
        features['dominant_frequency'] = 0
        features['dominant_amplitude'] = 0
    
    # 9. Top-3 Peak Frequencies
    if len(peaks) >= 3:
        top3_idx = peaks[np.argsort(magnitudes[peaks])[-3:]]
        features['peak_freq_1'] = freqs[top3_idx[-1]]
        features['peak_freq_2'] = freqs[top3_idx[-2]]
        features['peak_freq_3'] = freqs[top3_idx[-3]]
    elif len(peaks) > 0:
        for i in range(3):
            if i < len(peaks):
                features[f'peak_freq_{i+1}'] = freqs[peaks[i]]
            else:
                features[f'peak_freq_{i+1}'] = 0
    else:
        features['peak_freq_1'] = 0
        features['peak_freq_2'] = 0
        features['peak_freq_3'] = 0
    
    # 10. Bandpower features
    band1_mask = (freqs >= 0) & (freqs < 5)
    band2_mask = (freqs >= 5) & (freqs < 10)
    features['bandpower_0_5hz'] = np.sum(magnitudes[band1_mask] ** 2)
    features['bandpower_5_10hz'] = np.sum(magnitudes[band2_mask] ** 2)
    
    # 11. Periodicity (autocorrelation-based)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    if len(autocorr) > 1:
        features['periodicity'] = autocorr[1] / autocorr[0] if autocorr[0] != 0 else 0
    else:
        features['periodicity'] = 0
    
    # 12. Fundamental Frequency (lowest significant peak)
    if len(peaks) > 0:
        features['fundamental_frequency'] = freqs[peaks[0]]
    else:
        features['fundamental_frequency'] = 0
    
    # 13. Harmonic Ratio
    if len(peaks) > 0:
        harmonic_power = np.sum(magnitudes[peaks] ** 2)
        total_power = np.sum(magnitudes ** 2)
        features['harmonic_ratio'] = harmonic_power / total_power if total_power > 0 else 0
    else:
        features['harmonic_ratio'] = 0
    
    return features
```

**Output:**
- Save as: `data/08_spectral/SPECTRAL_FEATURES.csv`
- Include metadata columns: `subject_id`, `activity_label`, `sensor_type`, `device`
- Generate visualization: histogram/boxplot of spectral features

---

### **Task 2: Apply Best Scaling Method**

Create: `step9_spectral_scaling.py`

**Requirements:**
1. Based on previous results, the **best scaler is MinMax** (achieved 72.21% accuracy)
2. Apply MinMaxScaler to all spectral features
3. Compare before/after statistics (mean, std, min, max)
4. Generate visualizations showing scaling effect

**Justification to include:**
> "MinMax scaling was chosen because it achieved the highest accuracy (72.21%) with RandomForest in time-domain features. MinMax preserves the original distribution shape while ensuring all features are in [0,1] range, which is beneficial for distance-based algorithms and prevents features with larger magnitudes from dominating."

**Output:**
- Save as: `data/09_spectral_scaled/SCALED_SPECTRAL_FEATURES.csv`
- Save comparison stats: `scaling_spectral_comparison_stats.csv`
- Generate visualizations in `visualizations/spectral_scaling/`

---

### **Task 3: Apply Best Feature Selection**

Create: `step10_spectral_selection.py`

**Requirements:**
1. Based on previous work, use **Variance Threshold + Mutual Information** (same as Step 5)
2. Apply to spectral features
3. Select top 30 features (or adjust based on number of spectral features)
4. Report feature importance ranking

**Justification to include:**
> "Variance Threshold + Mutual Information was used successfully in time-domain feature selection. This two-stage approach first removes low-variance features (noise/constants), then ranks remaining features by their mutual information with the target class, selecting the most discriminative features."

**Output:**
- Save as: `data/10_spectral_selected/FINAL_SELECTED_SPECTRAL_FEATURES.csv`
- Save feature list: `selected_spectral_features.txt`
- Generate MI importance plot in `visualizations/spectral_selection/`

---

### **Task 4: Train ML Models on Spectral Features**

Create: `step11_spectral_model_training.py`

**Requirements:**
1. Train the same 4 models on spectral features:
   - K-Nearest Neighbors (k=5)
   - Naive Bayes (GaussianNB)
   - Decision Tree (max_depth=20)
   - Random Forest (n_estimators=100, max_depth=20)

2. Use 80-20 train-test split with stratification
3. Compute metrics for each model:
   - Accuracy
   - Precision (macro & weighted)
   - Recall (macro & weighted)
   - F1-score (macro & weighted)

4. Generate confusion matrices for all models
5. Create comparison table and bar chart

**Output:**
- Save results: `data/11_spectral_results/spectral_model_results.csv`
- Save confusion matrices: `visualizations/spectral_confusion_matrices/`
- Save comparison plot: `visualizations/spectral_model_comparison.png`

---

### **Task 5: Combined Results & Interpretation**

Create: `step12_final_comparison.py`

**Requirements:**
1. Load both time-domain results (from step6) and spectral results (from step11)
2. Create comprehensive comparison table
3. Generate combined visualizations
4. Write interpretation summary

**Comparison Table Format:**
```
| Feature Type | Model | Accuracy | Precision | Recall | F1-score |
|--------------|-------|----------|-----------|--------|----------|
| Time-Domain  | RF    | 72.21%   | ...       | ...    | ...      |
| Spectral     | RF    | ??%      | ...       | ...    | ...      |
| ...          | ...   | ...      | ...       | ...    | ...      |
```

**Interpretation Questions to Answer:**
1. Which scaling technique worked best and why?
2. Which feature-selection method was most effective?
3. Which classifier performed best on spectral features?
4. Do spectral features improve performance compared to time-domain features?
5. Which feature type (time vs spectral) is more discriminative for HAR?

**Output:**
- Save as: `data/12_final/combined_results.csv`
- Generate: `FINAL_REPORT.md` with all findings
- Update main `README.md` with spectral results

---

## CODE STRUCTURE GUIDELINES

### File Naming Convention
Follow existing pattern:
- `step8_spectral_features.py`
- `step9_spectral_scaling.py`
- `step10_spectral_selection.py`
- `step11_spectral_model_training.py`
- `step12_final_comparison.py`

### Each file should include:
1. Module docstring explaining the step
2. Import statements (follow existing style)
3. Helper functions with docstrings
4. Main `run()` function that can be called from pipeline
5. `if __name__ == '__main__':` block
6. Logging using the logger utility
7. Progress indicators for long operations

### Example Template:
```python
"""
WISDM-51 Activity Recognition Pipeline
Step X: [Description]

[Detailed explanation of what this step does]
"""

import os
import numpy as np
import pandas as pd
from config import DATA_DIR, VIS_DIR, METADATA_COLS
from logger import logger

def helper_function(data):
    """Helper function description."""
    # implementation
    pass

def run(input_df=None):
    """Execute Step X: [Description]."""
    logger.header("STEP X: [Title]")
    
    # Load data if not provided
    if input_df is None:
        input_df = pd.read_csv(os.path.join(DATA_DIR, 'path/to/input.csv'))
    
    logger.log(f"Processing {len(input_df):,} samples...")
    
    # Main processing logic
    result_df = process_data(input_df)
    
    # Save output
    output_dir = os.path.join(DATA_DIR, 'XX_output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output.csv')
    result_df.to_csv(output_path, index=False)
    
    logger.log(f"Saved to {output_path}")
    logger.log(f"Final shape: {result_df.shape}")
    logger.log("")
    
    return result_df

if __name__ == '__main__':
    run()
```

---

## CONFIGURATION UPDATES

Add to `config.py`:

```python
# Spectral feature parameters
SPECTRAL_ROLLOFF_THRESHOLD = 0.85  # 85% energy threshold
SPECTRAL_PEAK_PROMINENCE = 0.5     # Relative to mean magnitude
NUM_SPECTRAL_FEATURES = 30         # Features to select

# Frequency bands (Hz)
FREQ_BANDS = {
    'low': (0, 5),
    'mid': (5, 10),
    'high': (10, 10)  # Nyquist frequency = sampling_rate/2 = 10Hz
}
```

---

## VISUALIZATION REQUIREMENTS

### For Spectral Features (Step 8):
1. **Spectral Feature Distribution**
   - 10×12 grid of histograms for top 117 features
   - Similar style to existing `raw_features_histogram.png`

2. **Sample Spectrum Plot**
   - Show frequency spectrum for 3 sample windows (one per activity type: static, dynamic, transition)
   - Include magnitude vs frequency plot

### For Scaling (Step 9):
1. **Before/After Boxplots**
   - Compare raw vs scaled spectral features
   - Show first 15 features

2. **Before/After Histograms**
   - Overlay raw and scaled distributions

### For Feature Selection (Step 10):
1. **MI Importance Bar Chart**
   - Top 30 spectral features by mutual information
   - Horizontal bars, sorted descending

### For Model Training (Step 11):
1. **Confusion Matrices** (4 total)
   - One per model (KNN, NB, DT, RF)
   - 18×18 heatmap with activity labels
   - Include accuracy in title

2. **Model Comparison Bar Chart**
   - Grouped bars showing accuracy, precision, recall, F1
   - All 4 models on same plot

### For Final Comparison (Step 12):
1. **Time vs Spectral Performance**
   - Side-by-side bar chart comparing best models
   - Include error bars if available

2. **Feature Importance Comparison**
   - Show top 10 time-domain vs top 10 spectral features

---

## TESTING & VALIDATION

Before considering work complete, verify:

- [ ] All CSV files saved with correct naming
- [ ] All visualizations generated and saved
- [ ] No NaN/Inf values in processed data
- [ ] Feature counts match expected (117 spectral features initially)
- [ ] Model results reasonable (accuracy > 50% for best model)
- [ ] Confusion matrices show diagonal dominance
- [ ] All paths use `os.path.join()` for cross-platform compatibility
- [ ] Logger outputs informative messages
- [ ] Code follows existing style conventions
- [ ] Files can run independently and as part of pipeline

---

## FINAL DELIVERABLES CHECKLIST

### Code Files:
- [ ] `step8_spectral_features.py`
- [ ] `step9_spectral_scaling.py`
- [ ] `step10_spectral_selection.py`
- [ ] `step11_spectral_model_training.py`
- [ ] `step12_final_comparison.py`
- [ ] Updated `config.py`
- [ ] Updated `run_pipeline.py` to include steps 8-12

### Data Files:
- [ ] `data/08_spectral/SPECTRAL_FEATURES.csv`
- [ ] `data/09_spectral_scaled/SCALED_SPECTRAL_FEATURES.csv`
- [ ] `data/09_spectral_scaled/scaling_spectral_comparison_stats.csv`
- [ ] `data/10_spectral_selected/FINAL_SELECTED_SPECTRAL_FEATURES.csv`
- [ ] `data/10_spectral_selected/selected_spectral_features.txt`
- [ ] `data/11_spectral_results/spectral_model_results.csv`
- [ ] `data/12_final/combined_results.csv`

### Visualizations:
- [ ] `visualizations/spectral_features/` (feature distributions)
- [ ] `visualizations/spectral_scaling/` (before/after comparison)
- [ ] `visualizations/spectral_selection/` (MI importance)
- [ ] `visualizations/spectral_confusion_matrices/` (4 confusion matrices)
- [ ] `visualizations/spectral_model_comparison.png`
- [ ] `visualizations/final_comparison.png`

### Documentation:
- [ ] `FINAL_REPORT.md` (comprehensive findings)
- [ ] Updated `README.md` (include spectral results)
- [ ] `interpretation_summary.txt` (1-2 paragraphs)

---

## EXECUTION ORDER

Run in this sequence:

```bash
# New spectral feature pipeline
python step8_spectral_features.py
python step9_spectral_scaling.py
python step10_spectral_selection.py
python step11_spectral_model_training.py
python step12_final_comparison.py

# Or run all at once (after updating run_pipeline.py):
python run_pipeline.py from 8
```

---

## EXPECTED OUTCOMES

### Performance Expectations:
- Spectral features should achieve **60-80% accuracy** (comparable or better than time-domain)
- RandomForest likely to perform best (as with time-domain)
- Some activities (static: sitting, standing) may show better discrimination with spectral features
- Dynamic activities (walking, jogging) may perform similarly with both feature types

### Common Pitfalls to Avoid:
1. **FFT window length**: Use exactly 60 samples (matches window size)
2. **Frequency range**: Nyquist frequency is 10 Hz (sampling_rate/2)
3. **Zero-division**: Always check denominators before division
4. **Feature naming**: Use consistent naming with axis suffix (_x, _y, _z)
5. **Memory**: Process in batches if dataset is large (>100k windows)

---

## SUPPORT INFORMATION

### Dataset Details:
- **Sampling Rate**: 20 Hz
- **Window Size**: 60 samples (3 seconds)
- **Overlap**: 50% (hop size = 30)
- **Activities**: 18 classes (A-S, excluding N)
- **Sensors**: Accelerometer + Gyroscope
- **Devices**: Phone + Watch
- **Subjects**: 51 participants (IDs 1600-1650)

### Key References:
- **Scipy FFT**: https://docs.scipy.org/doc/scipy/reference/fft.html
- **Signal Processing**: https://docs.scipy.org/doc/scipy/reference/signal.html
- **Sklearn Preprocessing**: https://scikit-learn.org/stable/modules/preprocessing.html

---

## SUCCESS CRITERIA

Your implementation is successful when:

1. ✅ All 117 spectral features extracted correctly
2. ✅ Scaling improves model performance
3. ✅ Feature selection reduces dimensionality while maintaining accuracy
4. ✅ All 4 models trained and evaluated
5. ✅ Results documented with clear interpretation
6. ✅ Visualizations are clear and informative
7. ✅ Code is clean, documented, and follows existing patterns
8. ✅ Pipeline runs end-to-end without errors

---

## FINAL NOTES

- **Code Quality**: Follow PEP 8, add docstrings, handle edge cases
- **Efficiency**: Use vectorized operations where possible (numpy/pandas)
- **Reproducibility**: Set random_state=42 for all random operations
- **Documentation**: Comment complex logic, explain parameter choices
- **Error Handling**: Use try-except for file I/O and external operations

Good luck! This completes the WISDM-51 HAR project with both time-domain and frequency-domain feature analysis.
