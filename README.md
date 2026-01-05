# WISDM-51 Activity Recognition Pipeline

## Overview
Machine learning pipeline for human activity recognition using the WISDM-51 dataset from UCI Machine Learning Repository.

## Dataset
- **Source**: WISDM-51 (Wireless Sensor Data Mining)
- **Subjects**: 51 participants (IDs 1600-1650)
- **Activities**: 18 different activities
- **Sensors**: Phone and Watch (accelerometer + gyroscope)
- **Sampling Rate**: 20 Hz

## Pipeline Steps
1. **Data Cleaning** - Load and clean raw sensor data
2. **Windowing** - Create 3-second sliding windows (60 samples, 50% overlap)
3. **Feature Extraction** - Extract 60 time-domain features per window
4. **Scaling** - Apply MinMax, Standard, and Robust scaling
5. **Feature Selection** - Variance threshold + mutual information (top 30 features)
6. **Model Training** - Train KNN, Naive Bayes, Decision Tree, Random Forest
7. **Results Summary** - Aggregate and report results

## Results

### Best Model
| Scaler | Model | Accuracy | Macro F1 |
|--------|-------|----------|----------|
| minmax | RandomForest | 72.21% | 72.10% |

### All Results (Sorted by Accuracy)
| Rank | Scaler | Model | Accuracy |
|------|--------|-------|----------|
| 1 | minmax | RandomForest | 72.21% |
| 2 | robust | RandomForest | 67.31% |
| 3 | standard | RandomForest | 67.22% |
| 4 | standard | KNN | 66.48% |
| 5 | robust | KNN | 65.74% |
| 6 | minmax | DecisionTree | 57.85% |
| 7 | robust | DecisionTree | 56.25% |
| 8 | standard | DecisionTree | 56.21% |
| 9 | minmax | KNN | 55.70% |
| 10 | minmax | NaiveBayes | 19.16% |
| 11 | standard | NaiveBayes | 13.76% |
| 12 | robust | NaiveBayes | 13.76% |

## Usage

### Run Complete Pipeline
```bash
python run_pipeline.py
```

### Run Individual Steps
```bash
python step1_data_cleaning.py
python step2_windowing.py
python step3_feature_extraction.py
python step4_scaling.py
python step5_feature_selection.py
python step6_model_training.py
python step7_results_summary.py
```

## Directory Structure
```
├── data/
│   ├── 01_cleaned/         # Cleaned sensor data
│   ├── 02_windowed/        # Windowed data
│   ├── 03_features/        # Extracted features
│   ├── 04_scaled/          # Scaled features
│   ├── 05_selected/        # Selected features
│   └── 06_results/         # Model results
├── visualizations/
│   ├── confusion_matrices/ # Model confusion matrices
│   ├── feature_distributions/ # Feature visualizations
│   ├── feature_selection/  # Feature importance plots
│   └── scaling_comparison/ # Before/after scaling plots
├── raw/                    # Original WISDM-51 data
└── *.py                    # Pipeline scripts
```

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

---
*Last updated: 2026-01-05 13:32:27*
