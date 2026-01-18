# WISDM-51 Activity Recognition Pipeline

## Overview
Machine learning pipeline for human activity recognition using the WISDM-51 dataset, 
comparing time-domain and spectral (frequency-domain) feature approaches.

## Dataset
- **Source**: WISDM-51 (UCI ML Repository)
- **Subjects**: 51 participants (IDs 1600-1650)
- **Activities**: 18 different activities
- **Sensors**: Phone and Watch (accelerometer + gyroscope)
- **Sampling Rate**: 20 Hz

## Pipeline Steps

### Part 1: Time-Domain Features (Steps 1-7)
1. **Data Cleaning** - Load and clean raw sensor data
2. **Windowing** - 3-second windows (60 samples, 50% overlap)
3. **Feature Extraction** - 60 time-domain features per window
4. **Scaling** - MinMax, Standard, Robust scaling
5. **Feature Selection** - Variance threshold + Mutual Information
6. **Model Training** - KNN, Naive Bayes, Decision Tree, Random Forest
7. **Results Summary** - Aggregate and report results

### Part 2: Spectral Features (Steps 8-12)
8. **Spectral Features** - FFT-based frequency-domain feature extraction
9. **Spectral Scaling** - MinMax scaling (best from Part 1)
10. **Spectral Selection** - Variance threshold + Mutual Information
11. **Spectral Training** - Train same 4 models on spectral features
12. **Final Comparison** - Compare time-domain vs spectral results

## Results Summary

### Best Overall Result
| Feature Type | Scaler | Model | Accuracy |
|--------------|--------|-------|----------|
| Time-Domain | minmax | RandomForest | **72.21%** |

### Time-Domain Best
| Scaler | Model | Accuracy | Macro F1 |
|--------|-------|----------|----------|
| minmax | RandomForest | 72.21% | 72.10% |

### Spectral Best
| Scaler | Model | Accuracy | Macro F1 |
|--------|-------|----------|----------|
| MinMax | RandomForest | 44.69% | 43.70% |

### Complete Results
| Rank | Feature Type | Scaler | Model | Accuracy |
|------|--------------|--------|-------|----------|
| 1 | Time-Domain | minmax | RandomForest | 72.21% |
| 2 | Time-Domain | robust | RandomForest | 67.31% |
| 3 | Time-Domain | standard | RandomForest | 67.22% |
| 4 | Time-Domain | standard | KNN | 66.48% |
| 5 | Time-Domain | robust | KNN | 65.74% |
| 6 | Time-Domain | minmax | DecisionTree | 57.85% |
| 7 | Time-Domain | robust | DecisionTree | 56.25% |
| 8 | Time-Domain | standard | DecisionTree | 56.21% |
| 9 | Time-Domain | minmax | KNN | 55.70% |
| 10 | Spectral | minmax | RandomForest | 44.69% |

## Usage

### Run Complete Pipeline
```bash
python run_pipeline.py
```

### Run Individual Steps
```bash
# Time-domain pipeline
python step1_data_cleaning.py
python step2_windowing.py
python step3_feature_extraction.py
python step4_scaling.py
python step5_feature_selection.py
python step6_model_training.py
python step7_results_summary.py

# Spectral pipeline
python step8_spectral_features.py
python step9_spectral_scaling.py
python step10_spectral_selection.py
python step11_spectral_model_training.py
python step12_final_comparison.py
```

### Run from specific step
```bash
python run_pipeline.py from 8  # Run spectral pipeline only
```

## Directory Structure
```
├── data/
│   ├── 01_cleaned/              # Cleaned sensor data
│   ├── 02_windowed/             # Windowed data
│   ├── 03_features/             # Time-domain features
│   ├── 04_scaled/               # Scaled time-domain features
│   ├── 05_selected/             # Selected time-domain features
│   ├── 06_results/              # Time-domain model results
│   ├── 08_spectral/             # Spectral features
│   ├── 09_spectral_scaled/      # Scaled spectral features
│   ├── 10_spectral_selected/    # Selected spectral features
│   ├── 11_spectral_results/     # Spectral model results
│   └── 12_final/                # Combined results
├── visualizations/
│   ├── confusion_matrices/      # Time-domain confusion matrices
│   ├── spectral_confusion_matrices/  # Spectral confusion matrices
│   ├── spectral_features/       # Spectral feature distributions
│   └── ...
├── raw/                         # Original WISDM-51 data
├── FINAL_REPORT.md              # Comprehensive analysis report
└── *.py                         # Pipeline scripts
```

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

---
*Last updated: 2026-01-18 21:33:04*
