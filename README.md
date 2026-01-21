# WISDM-51 Activity Recognition Pipeline

## Overview
Optimized machine learning pipeline for human activity recognition using the WISDM-51 dataset with combined time-domain, frequency-domain, and advanced features. Achieves **78.26% accuracy** using ensemble methods.

## Dataset
- **Source**: WISDM-51 (UCI ML Repository)
- **Subjects**: 51 participants (IDs 1600-1650)
- **Activities**: 18 different activities
- **Sensors**: Phone and Watch (accelerometer + gyroscope)
- **Sampling Rate**: 20 Hz
- **Windows**: 516,094 total (10-second windows, 50% overlap)
- **Split**: 412,875 train / 103,219 test samples

## Optimized Pipeline Architecture

The pipeline combines three feature extraction approaches for maximum accuracy:

### 1. Data Preparation (Steps 1-2)
- **Step 1: Data Cleaning** - Load and clean raw sensor data
- **Step 2: Windowing** - 10-second windows (200 samples, 50% overlap)

### 2. Feature Engineering (Steps 3, 3b, 8, 3c)
- **Step 3: Basic Features** - Statistical features (mean, std, min, max, etc.)
- **Step 3b: Advanced Features** - Wavelet coefficients, entropy, jerk metrics
- **Step 8: Spectral Features** - FFT-based frequency-domain features
- **Step 3c: Feature Combination** - Merge all feature types into unified dataset

### 3. Preprocessing (Steps 4-5)
- **Step 4: Scaling** - MinMax normalization on combined features
- **Step 5: Feature Selection** - SelectKBest with mutual information

### 4. Model Training (Steps 6b-6c)
- **Step 6b: Optimized Models** - Hyperparameter tuning with RandomizedSearchCV
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Gradient Boosting (with early stopping)
- **Step 6c: Ensemble Models** - Combining optimized base models
  - Hard Voting Ensemble
  - Soft Voting Ensemble  
  - Stacking Ensemble (with LogisticRegression meta-learner)

## Performance Results

### 🏆 Best Model: Stacking Ensemble - 78.26%

| Rank | Model | Test Accuracy | Notes |
|------|-------|--------------|-------|
| 1 | **Stacking Ensemble** | **78.26%** | Best overall - combines KNN, DecisionTree, RandomForest |
| 2 | Random Forest | 77.44% | Strong individual performer |
| 3 | Hard Voting | 75.05% | Majority vote ensemble |
| 4 | Soft Voting | 70.85% | Probability-based ensemble |
| 5 | K-Nearest Neighbors | 69.94% | Distance-based classification |
| 6 | Decision Tree | 62.32% | Fast but less accurate |
| 7 | Gradient Boosting | 51.63% | Trade-off: speed over accuracy |

### Optimization Impact
- **Runtime**: Reduced from 5+ hours to **3h 22min** (2.5x speedup)
- **Accuracy**: Improved from 72.21% baseline to **78.26%** (+6.05%)
- **Key Optimizations**:
  - Subsampled hyperparameter tuning (30K samples instead of 412K)
  - Reduced cross-validation folds (3 instead of 5)
  - GradientBoosting early stopping (10 iterations no improvement)
  - RemoOptimized Pipeline (Recommended)
```bash
# Activate virtual environment
source .venv/bin/activate

# Run complete optimized pipeline (~3h 22min)
python run_pipeline.py optimized
```

### Run Individual Steps
```bash
# Data preparation
python run_pipeline.py step 1    # Data cleaning
python run_pipeline.py step 2    # Windowing

# Feature extraction
python run_pipeline.py step 3    # Basic features
python run_pipeline.py step 8    # Spectral features

# Model training
python run_pipeline.py step 6b   # Optimized individual models
python run_pipeline.py step 6c   # Ensemble models
```

### Available Steps
Valid steps: `1, 2, 3, 4, 5, 8, 6b, 6c`

**Note**: Steps 3b and 3c are automatically executed within the optimized pipeline.

### Run from specific step
```bash
python run_pipeline.py from 8  # Run spectral pipeline only
```

## Directory Structure
```
WISDM-51_Project/
├── data/                        # Generated during pipeline execution
│   ├── 01_cleaned/              # Cleaned sensor data
│   ├── 02_windowed/             # 10-second windowed data
│   ├── 03_features/             # Basic statistical features
│   ├── 03b_advanced/            # Advanced features (wavelet, entropy, jerk)
│   ├── 03c_combined/            # All features combined
│   ├── 04_scaled/               # MinMax scaled features
│   ├── 05_selected/             # Selected features (SelectKBest)
│   ├── 06b_optimized_results/   # Individual model results + trained models
│   ├── 06c_ensemble_results/    # Ensemble model results + trained models
│   └── 08_spectral_features/    # FFT frequency-domain features
├── visualizations/              # Generated confusion matrices and charts
│   ├── optimized_confusion_matrices/
│   └── ensemble_confusion_matrices/
├── raw/                         # Original WISDM-51 dataset
│   ├── phone/
│   │   ├── accel/               # Phone accelerometer data
│   │   └── gyro/                # Phone gyroscope data
│   └── watch/                   # Watch sensor data
├── .venv/                       # Python virtual environment
├── .cache/                      # Cache for resumable pipeline execution
├── run_pipeline.py              # Main pipeline orchestrator
├── step1_data_cleaning.py       # Data cleaning
├── step2_windowing.py           # Window creation
├── step3_feature_extraction.py  # Basic features
├── step3b_advanced_features.py  # Advanced features
├── step3c_combine_features.py   # Feature combination
├── step4_scaling.py             # Feature scaling
├── step5_feature_selection.py   # Feature selection
├── step6b_optimized_models.py   # Optimized model training
├── step6c_ensemble_models.py    # Ensemble model training
├── step8_spectral_features.py   # Spectral feature extraction
├── cache_utils.py               # Caching utilities
├── config.py                    # Pipeline configuration
├── logger.py                    # Logging utilities
├── activity_key.txt             # Activity label mappings
└── README.md                    # This file
```

## Requirements
```bash
# Python 3.13+ recommended (tested on 3.13.3)
pip install pandas numpy scikit-learn matplotlib seaborn scipy pywavelets tqdm
```

## Key Features
- ✅ Combined feature engineering (time + frequency + advanced)
- ✅ Optimized hyperparameter tuning with subsampling
- ✅ Ensemble methods for improved accuracy
- ✅ Early stopping for GradientBoosting
- ✅ Caching for pipeline resumability
- ✅ Parallel processing for faster training
- ✅ Comprehensive visualization and logging

## Expected Runtime
- **Full Optimized Pipeline**: ~3 hours 22 minutes
  - Data preparation: ~5 min
  - Feature extraction: ~25 min
  - Step 6b (4 models): ~2h 31min
  - Step 6c (3 ensembles): ~45 min

---
*Last updated: 2026-01-21*
