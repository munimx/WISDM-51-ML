# WISDM-51 Activity Recognition - Final Report

## Executive Summary

This report presents the complete analysis of human activity recognition using the WISDM-51 dataset, 
comparing **time-domain** and **spectral (frequency-domain)** feature approaches.

**Generated:** 2026-01-18 21:33:04

---

## 1. Dataset Overview

- **Dataset:** WISDM-51 (Wireless Sensor Data Mining)
- **Subjects:** 51 participants (IDs 1600-1650)
- **Activities:** 18 different activities (Walking, Jogging, Stairs, Sitting, Standing, etc.)
- **Sensors:** Accelerometer + Gyroscope
- **Devices:** Smartphone + Smartwatch
- **Sampling Rate:** 20 Hz
- **Window Size:** 3 seconds (60 samples)
- **Window Overlap:** 50%

---

## 2. Methodology

### 2.1 Time-Domain Features (Steps 1-7)
- **60 features** extracted per window (20 per axis)
- Features include: mean, std, min, max, range, median, MAD, IQR, skewness, kurtosis, 
  energy, RMS, zero crossings, Hjorth parameters, peak count, autocorrelation

### 2.2 Spectral Features (Steps 8-11)
- **39 features** extracted per window (13 per axis)
- Features include: spectral energy, entropy, centroid, spread, flux, roll-off, flatness,
  dominant frequency, bandpower (0-5Hz, 5-10Hz), periodicity, harmonic ratio

### 2.3 Scaling
- **Time-Domain:** MinMax, Standard, and Robust scaling compared
- **Spectral:** MinMax scaling (best performer from time-domain)

### 2.4 Feature Selection
- **Method:** Variance Threshold (0.01) + Mutual Information
- **Selection:** Top 30 features (or fewer if not available)

### 2.5 Models
- K-Nearest Neighbors (k=5)
- Gaussian Naive Bayes
- Decision Tree (max_depth=20)
- Random Forest (n_estimators=100, max_depth=20)

---

## 3. Results

### 3.1 Best Overall Result
| Feature Type | Scaler | Model | Accuracy | Macro F1 |
|--------------|--------|-------|----------|----------|
| Time-Domain | minmax | RandomForest | **72.21%** | 72.10% |

### 3.2 Time-Domain Results (Best per Model)
| Model | Best Scaler | Accuracy | Precision | Recall | F1-Score |
|-------|-------------|----------|-----------|--------|----------|
| RandomForest | minmax | 72.21% | 72.49% | 72.20% | 72.10% |
| KNN | standard | 66.48% | 66.53% | 66.47% | 66.19% |
| DecisionTree | minmax | 57.85% | 58.43% | 57.84% | 58.02% |
| NaiveBayes | minmax | 19.16% | 20.54% | 19.12% | 17.67% |

### 3.3 Spectral Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| RandomForest | 44.69% | 44.41% | 44.61% | 43.70% |
| DecisionTree | 33.58% | 33.73% | 33.54% | 33.51% |
| KNN | 33.23% | 32.85% | 33.18% | 32.22% |
| NaiveBayes | 18.52% | 17.88% | 18.53% | 15.72% |

### 3.4 Comparison Summary
| Feature Type | Best Model | Best Accuracy |
|--------------|------------|---------------|
| Time-Domain | RandomForest (minmax) | 72.21% |
| Spectral | RandomForest | 44.69% |

---

## 4. Analysis and Interpretation

### 4.1 Which scaling technique worked best and why?
**MinMax scaling** achieved the best results with time-domain features (72.21% with RandomForest).
MinMax preserves the original distribution shape while normalizing features to [0,1], which:
- Prevents features with larger magnitudes from dominating
- Works well with distance-based algorithms (KNN) and tree-based methods
- Maintains relative relationships between feature values

### 4.2 Which feature-selection method was most effective?
**Variance Threshold + Mutual Information** proved effective:
- Variance threshold removes constant/near-constant features (noise)
- Mutual Information identifies features with high discriminative power
- This two-stage approach balances computational efficiency with selection quality

### 4.3 Which classifier performed best?
**Random Forest** consistently outperformed other classifiers:
- Handles non-linear relationships well
- Robust to overfitting through ensemble averaging
- Effectively captures complex patterns in sensor data
- KNN performed second-best, benefiting from scaled features

### 4.4 Do spectral features improve performance?
Based on results:
- Time-Domain Best: 72.21%
- Spectral Best: 44.69%
- Difference: 27.52%

Time-domain features outperformed spectral features.

### 4.5 Which feature type is more discriminative for HAR?
**Time-domain features** capture:
- Statistical properties (mean, variance, range)
- Signal morphology (peaks, zero crossings)
- Temporal patterns (autocorrelation)

**Spectral features** capture:
- Frequency content (dominant frequencies)
- Signal complexity (entropy, flatness)
- Energy distribution (bandpower)

For HAR, both feature types provide complementary information. Time-domain features 
excel at capturing movement patterns, while spectral features identify repetitive 
motion characteristics.

---

## 5. Conclusions

1. **Best Configuration:** Time-Domain + minmax + RandomForest (72.21%)

2. **Scaling:** MinMax scaling provides consistent improvements across models

3. **Feature Selection:** Variance + MI effectively reduces dimensionality while preserving discriminative power

4. **Model Choice:** Random Forest is the most reliable classifier for this task

5. **Feature Engineering:** Both time-domain and spectral features provide valuable information for activity recognition

---

## 6. Files Generated

### Data Files:
- `data/01_cleaned/cleaned_data.csv` - Cleaned sensor data
- `data/02_windowed/windowed_data.csv` - Windowed data
- `data/03_features/features_raw.csv` - Time-domain features
- `data/04_scaled/` - Scaled time-domain features
- `data/05_selected/` - Selected time-domain features
- `data/06_results/model_results.csv` - Time-domain model results
- `data/08_spectral/SPECTRAL_FEATURES.csv` - Spectral features
- `data/09_spectral_scaled/SCALED_SPECTRAL_FEATURES.csv` - Scaled spectral features
- `data/10_spectral_selected/FINAL_SELECTED_SPECTRAL_FEATURES.csv` - Selected spectral features
- `data/11_spectral_results/spectral_model_results.csv` - Spectral model results
- `data/12_final/combined_results.csv` - Combined comparison results

### Visualizations:
- `visualizations/confusion_matrices/` - Time-domain confusion matrices
- `visualizations/spectral_confusion_matrices/` - Spectral confusion matrices
- `visualizations/spectral_model_comparison.png` - Spectral model comparison
- `visualizations/final_comparison.png` - Time vs Spectral comparison

---

*Report generated automatically by the WISDM-51 Activity Recognition Pipeline*
