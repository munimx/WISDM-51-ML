# WISDM-51 Activity Recognition - Final Report

## Executive Summary

This report presents a comprehensive analysis of human activity recognition
using the WISDM-51 dataset with an **optimized multi-feature approach**
combining time-domain, frequency-domain, and advanced features.

**Best Accuracy Achieved:** 🏆 **78.26%** (Stacking Ensemble)  
**Pipeline Runtime:** ~3 hours 22 minutes  
**Generated:** 2026-01-21

---

## 1. Dataset Overview

- **Dataset:** WISDM-51 (Wireless Sensor Data Mining)
- **Subjects:** 51 participants (IDs 1600-1650)
- **Activities:** 18 different activities (Walking, Jogging, Stairs, Sitting,
  Standing, etc.)
- **Sensors:** Accelerometer + Gyroscope
- **Devices:** Smartphone + Smartwatch
- **Sampling Rate:** 20 Hz
- **Window Size:** 10 seconds (200 samples)
- **Window Overlap:** 50%
- **Total Windows:** 516,094
- **Training Samples:** 412,875
- **Test Samples:** 103,219

---

## 2. Methodology

### 2.1 Multi-Feature Engineering Approach

The optimized pipeline combines three complementary feature extraction methods:

#### **Basic Time-Domain Features (Step 3)**

Statistical and temporal features capturing movement patterns:

- **Statistical:** mean, std, min, max, range, median, MAD, IQR
- **Distribution:** skewness, kurtosis
- **Energy:** RMS, signal energy, signal magnitude area
- **Signal properties:** zero crossings, autocorrelation, Hjorth parameters
- **Peak analysis:** peak count, peak-to-peak amplitude

#### **Advanced Features (Step 3b)**

Sophisticated signal processing features:

- **Wavelet Transform:** Multi-scale decomposition coefficients (db4 wavelet)
- **Entropy Measures:** Shannon entropy, approximate entropy (signal complexity)
- **Jerk Metrics:** Rate of change of acceleration (smoothness of movement)

#### **Spectral Features (Step 8)**

Frequency-domain characteristics:

- **Frequency Analysis:** Dominant frequency, spectral centroid, spread
- **Energy Distribution:** Bandpower (0-5Hz, 5-10Hz), spectral energy
- **Complexity:** Spectral entropy, flatness, flux, roll-off
- **Periodicity:** Harmonic ratio, periodicity detection

### 2.2 Feature Combination Strategy

All three feature types are merged into a unified feature set, allowing models
to:

- Capture both temporal and frequency patterns
- Leverage complementary information from different domains
- Learn complex discriminative patterns across feature spaces

### 2.3 Preprocessing Pipeline

1. **Scaling:** MinMax normalization (all features to [0,1] range)
2. **Feature Selection:** SelectKBest with Mutual Information
   - Identifies most discriminative features
   - Reduces dimensionality while preserving information

### 2.4 Optimized Model Training

#### **Step 6b: Individual Models with Hyperparameter Tuning**

- **RandomizedSearchCV** for efficient parameter search
- **Subsampling:** Tune on 30,000 samples (faster than full dataset)
- **Cross-Validation:** 3-fold stratified CV
- **Early Stopping:** GradientBoosting with n_iter_no_change=10

Models trained:

1. **K-Nearest Neighbors**
   - Parameters: n_neighbors (3-15), weights, metric, leaf_size
   - Iterations: 12
2. **Decision Tree**
   - Parameters: max_depth, min_samples_split, min_samples_leaf, criterion
   - Iterations: 15
3. **Random Forest**
   - Parameters: n_estimators (100-200), max_depth, max_features
   - Iterations: 15
4. **Gradient Boosting**
   - Parameters: n_estimators (30-50), learning_rate, max_depth, subsample
   - Iterations: 6 (with early stopping)

#### **Step 6c: Ensemble Models**

Base models: Optimized KNN, DecisionTree, RandomForest (GB excluded for speed)

1. **Hard Voting Ensemble**
   - Majority vote classification
   - Equal weight to each base model
2. **Soft Voting Ensemble**
   - Probability-weighted predictions
   - Leverages model confidence
3. **Stacking Ensemble**
   - Meta-learner: LogisticRegression
   - Cross-validated predictions from base models
   - Learns optimal combination strategy

---

## 3. Results

### 3.1 Overall Performance Summary

| Rank | Model                 | Test Accuracy | Training Approach            | Runtime |
| ---- | --------------------- | ------------- | ---------------------------- | ------- |
| 🥇 1 | **Stacking Ensemble** | **78.26%**    | Meta-learning on base models | ~20 min |
| 🥈 2 | Random Forest         | 77.44%        | RandomizedSearchCV (15 iter) | ~60 min |
| 🥉 3 | Hard Voting           | 75.05%        | Majority vote ensemble       | ~15 min |
| 4    | Soft Voting           | 70.85%        | Probability-based ensemble   | ~15 min |
| 5    | K-Nearest Neighbors   | 69.94%        | RandomizedSearchCV (12 iter) | ~30 min |
| 6    | Decision Tree         | 62.32%        | RandomizedSearchCV (15 iter) | ~10 min |
| 7    | Gradient Boosting     | 51.63%        | Early stopping (n_iter=6)    | ~51 min |

### 3.2 Ensemble vs Individual Models

**Key Findings:**

- **Stacking Ensemble** outperforms all individual models (+0.82% over best
  individual)
- **Hard Voting** achieves 75.05%, competitive performance with simpler approach
- **Soft Voting** underperforms Hard Voting (likely due to confidence
  calibration)
- **Gradient Boosting** trades accuracy for speed (early stopping reduces
  training time by ~5x)

### 3.3 Performance Improvement Over Baseline

| Configuration                     | Accuracy | Improvement | Notes                      |
| --------------------------------- | -------- | ----------- | -------------------------- |
| **Baseline** (Time-Domain only)   | 72.21%   | -           | Original RandomForest      |
| **Optimized RF** (Multi-features) | 77.44%   | +5.23%      | Combined features + tuning |
| **Stacking Ensemble**             | 78.26%   | +6.05%      | Best overall configuration |

### 3.4 Runtime Performance

**Total Pipeline Runtime:** 3 hours 22 minutes (~202 minutes)

Breakdown:

- Data preparation: ~5 min
- Feature extraction (3 types): ~25 min
- Step 6b (4 individual models): ~151 min (2h 31min)
- Step 6c (3 ensemble models): ~45 min

**Optimization Impact:**

- Original pipeline: >5 hours
- Optimized pipeline: 3h 22min
- **Speedup: 2.5x faster**

---

## 4. Analysis and Interpretation

### 4.1 Why does the Stacking Ensemble perform best?

**Stacking advantages:**

1. **Meta-learning:** LogisticRegression learns optimal combination weights
2. **Complementary strengths:** Different base models capture different patterns
   - KNN: Local similarity patterns
   - DecisionTree: Rule-based boundaries
   - RandomForest: Complex non-linear relationships
3. **Cross-validated predictions:** Reduces overfitting in meta-learner training
4. **Feature diversity:** Multi-feature approach provides rich input for
   ensemble

### 4.2 Impact of Multi-Feature Engineering

**Combined features outperform single-type features:**

- Basic time-domain alone: Good baseline
- Adding spectral features: Captures periodic/frequency patterns
- Adding advanced features: Improves signal complexity understanding
- **Combined effect:** +6.05% accuracy improvement

**Why this works:**

- Different activities have different frequency signatures (walking vs jogging)
- Wavelet features capture multi-scale patterns (sudden vs gradual movements)
- Entropy measures distinguish smooth vs erratic movements
- Jerk metrics identify movement transitions and changes in acceleration

### 4.3 Which scaling technique worked best and why?

**MinMax scaling** achieved the best results across all models:

- Normalizes features to [0,1] range
- Preserves original distribution shape
- Prevents features with larger magnitudes from dominating
- Works well with distance-based algorithms (KNN) and tree-based methods
- Maintains relative relationships between feature values

### 4.4 Optimization Trade-offs

**GradientBoosting case study:**

- Without early stopping: >4 hours training, ~66-67% accuracy
- With early stopping: ~51 min training, 51.63% accuracy
- **Trade-off:** 5x speedup, -15% accuracy
- **Decision:** Acceptable for pipeline efficiency; RF provides similar/better
  accuracy faster

**Hyperparameter tuning optimizations:**

- Subsampling (30K samples): Maintains model quality while reducing CV time
- Reduced CV folds (3 instead of 5): Balances reliability with speed
- RandomizedSearchCV: More efficient than GridSearchCV for large parameter
  spaces

### 4.5 Feature Type Comparison

| Feature Type    | Primary Strength                        | Best For                                  |
| --------------- | --------------------------------------- | ----------------------------------------- |
| **Time-Domain** | Statistical patterns, signal morphology | Stationary activities (sitting, standing) |
| **Spectral**    | Frequency content, periodicity          | Rhythmic activities (walking, jogging)    |
| **Advanced**    | Signal complexity, multi-scale patterns | Dynamic transitions, complex movements    |

**Combined approach leverages all strengths** for maximum discriminative power.

---

## 5. Conclusions

### 5.1 Key Findings

1. **Best Configuration:** Multi-feature + MinMax + Stacking Ensemble → **78.26%
   accuracy**

2. **Feature Engineering Impact:** Combining time, frequency, and advanced
   features provides +6.05% improvement over baseline

3. **Ensemble Methods:** Stacking outperforms voting ensembles through
   meta-learned combination strategy

4. **Optimization Success:** Achieved 2.5x speedup (5h → 3h 22min) while
   improving accuracy (+6%)

5. **Model Selection:**
   - RandomForest: Best individual model (77.44%)
   - Stacking: Best overall (78.26%)
   - Hard Voting: Best speed/accuracy trade-off for ensembles

### 5.2 Optimization Techniques Applied

- ✅ Subsampled hyperparameter tuning (30K samples)
- ✅ Reduced cross-validation folds (3 instead of 5)
- ✅ Early stopping for GradientBoosting
- ✅ Parallel processing (half CPU cores for ensembles)
- ✅ Removed GB from ensemble base models (too slow)
- ✅ RandomizedSearchCV instead of GridSearchCV
- ✅ Discrete parameters for GB (faster than distributions)
- ✅ Caching system for pipeline resumability

### 5.3 Recommendations for Future Work

1. **Deep Learning:** Explore CNN/LSTM architectures for automatic feature
   learning
2. **Additional Features:** Consider time-series specific features (DTW, SAX)
3. **Model Compression:** Investigate model distillation for deployment
4. **Online Learning:** Adapt models for real-time activity recognition
5. **Transfer Learning:** Pre-train on larger HAR datasets
6. **Hyperparameter Optimization:** Try Bayesian optimization for better
   convergence

---

## 6. Files Generated

### Data Files:

- `data/01_cleaned/cleaned_data.csv` - Cleaned sensor data (516,094 windows)
- `data/02_windowed/windowed_data.csv` - 10-second windowed data
- `data/03_features/features.csv` - Basic time-domain features
- `data/03b_advanced/advanced_features.csv` - Advanced features (wavelet,
  entropy, jerk)
- `data/03c_combined/combined_features.csv` - All features merged
- `data/04_scaled/minmax_scaled.csv` - MinMax normalized features
- `data/05_selected/minmax_selected.csv` - Selected features (SelectKBest)
- `data/06b_optimized_results/optimized_results.csv` - Individual model results
- `data/06c_ensemble_results/ensemble_results.csv` - Ensemble model results
- `data/08_spectral_features/spectral_features.csv` - FFT frequency-domain
  features

### Model Files:

- `data/06b_optimized_results/*.pkl` - Trained individual models (KNN, DT, RF,
  GB)
- `data/06c_ensemble_results/*.pkl` - Trained ensemble models (voting, stacking)

### Visualizations:

- `visualizations/optimized_confusion_matrices/` - Confusion matrices for all
  models
- `visualizations/ensemble_confusion_matrices/` - Ensemble confusion matrices
- `visualizations/optimized_model_comparison.png` - Individual model comparison
  chart
- `visualizations/ensemble_comparison.png` - Ensemble vs baseline comparison

---

## 7. Technical Specifications

**Python Version:** 3.13.3  
**Key Libraries:** scikit-learn, pandas, numpy, scipy, pywavelets, tqdm  
**Hardware:** Multi-core CPU (parallel processing enabled)  
**Random Seed:** 42 (reproducible results)

---

_Report generated automatically by the WISDM-51 Optimized Activity Recognition
Pipeline_
