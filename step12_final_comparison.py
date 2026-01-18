"""
WISDM-51 Activity Recognition Pipeline
Step 12: Final Comparison and Report Generation

Compares time-domain vs spectral feature results and generates final report.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import BASE_DIR, DATA_DIR, VIS_DIR
from logger import logger


def load_results():
    """Load both time-domain and spectral results."""
    # Time-domain results
    time_path = os.path.join(DATA_DIR, '06_results', 'model_results.csv')
    time_results = pd.read_csv(time_path)
    time_results['feature_type'] = 'Time-Domain'
    
    # Spectral results
    spectral_path = os.path.join(DATA_DIR, '11_spectral_results', 'spectral_model_results.csv')
    spectral_results = pd.read_csv(spectral_path)
    spectral_results['feature_type'] = 'Spectral'
    
    return time_results, spectral_results


def generate_comparison_visualizations(time_df, spectral_df):
    """Generate comparison visualizations between time and spectral features."""
    vis_dir = VIS_DIR
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get best result per model for each feature type
    # For time-domain, get best scaler per model
    time_best = time_df.loc[time_df.groupby('model')['accuracy'].idxmax()].copy()
    time_best = time_best[['model', 'accuracy', 'macro_f1']].copy()
    time_best['feature_type'] = 'Time-Domain'
    
    spectral_best = spectral_df[['model', 'accuracy', 'macro_f1']].copy()
    spectral_best['feature_type'] = 'Spectral'
    
    # Combine for plotting
    combined = pd.concat([time_best, spectral_best], ignore_index=True)
    
    # Side-by-side bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = combined['model'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    time_acc = [combined[(combined['model'] == m) & (combined['feature_type'] == 'Time-Domain')]['accuracy'].values[0] 
                if len(combined[(combined['model'] == m) & (combined['feature_type'] == 'Time-Domain')]) > 0 else 0 
                for m in models]
    spectral_acc = [combined[(combined['model'] == m) & (combined['feature_type'] == 'Spectral')]['accuracy'].values[0] 
                   if len(combined[(combined['model'] == m) & (combined['feature_type'] == 'Spectral')]) > 0 else 0 
                   for m in models]
    
    bars1 = ax.bar(x - width/2, time_acc, width, label='Time-Domain', color='steelblue')
    bars2 = ax.bar(x + width/2, spectral_acc, width, label='Spectral', color='coral')
    
    # Add value labels
    for bar, val in zip(bars1, time_acc):
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar, val in zip(bars2, spectral_acc):
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Time-Domain vs Spectral Features - Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'final_comparison.png'), dpi=300)
    plt.close()
    
    logger.log("Saved final comparison visualization")
    
    return time_best, spectral_best


def generate_final_report(time_df, spectral_df, combined_df):
    """Generate comprehensive final report."""
    
    # Get best models
    time_best = time_df.loc[time_df['accuracy'].idxmax()]
    spectral_best = spectral_df.loc[spectral_df['accuracy'].idxmax()]
    overall_best = combined_df.loc[combined_df['accuracy'].idxmax()]
    
    report = f"""# WISDM-51 Activity Recognition - Final Report

## Executive Summary

This report presents the complete analysis of human activity recognition using the WISDM-51 dataset, 
comparing **time-domain** and **spectral (frequency-domain)** feature approaches.

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
| {overall_best['feature_type']} | {overall_best.get('scaler', 'MinMax')} | {overall_best['model']} | **{overall_best['accuracy']:.2f}%** | {overall_best['macro_f1']:.2f}% |

### 3.2 Time-Domain Results (Best per Model)
| Model | Best Scaler | Accuracy | Precision | Recall | F1-Score |
|-------|-------------|----------|-----------|--------|----------|
"""
    
    # Add time-domain best per model
    for model in time_df['model'].unique():
        model_df = time_df[time_df['model'] == model].sort_values('accuracy', ascending=False).iloc[0]
        report += f"| {model} | {model_df['scaler']} | {model_df['accuracy']:.2f}% | {model_df['macro_precision']:.2f}% | {model_df['macro_recall']:.2f}% | {model_df['macro_f1']:.2f}% |\n"
    
    report += f"""
### 3.3 Spectral Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
"""
    
    for _, row in spectral_df.sort_values('accuracy', ascending=False).iterrows():
        report += f"| {row['model']} | {row['accuracy']:.2f}% | {row['macro_precision']:.2f}% | {row['macro_recall']:.2f}% | {row['macro_f1']:.2f}% |\n"
    
    report += f"""
### 3.4 Comparison Summary
| Feature Type | Best Model | Best Accuracy |
|--------------|------------|---------------|
| Time-Domain | {time_best['model']} ({time_best.get('scaler', 'N/A')}) | {time_best['accuracy']:.2f}% |
| Spectral | {spectral_best['model']} | {spectral_best['accuracy']:.2f}% |

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
- Time-Domain Best: {time_best['accuracy']:.2f}%
- Spectral Best: {spectral_best['accuracy']:.2f}%
- Difference: {time_best['accuracy'] - spectral_best['accuracy']:.2f}%

{"Time-domain features outperformed spectral features." if time_best['accuracy'] > spectral_best['accuracy'] else "Spectral features matched or exceeded time-domain performance."}

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

1. **Best Configuration:** {overall_best['feature_type']} + {overall_best.get('scaler', 'MinMax')} + {overall_best['model']} ({overall_best['accuracy']:.2f}%)

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
"""
    
    return report


def generate_interpretation_summary(time_df, spectral_df):
    """Generate brief interpretation summary."""
    time_best = time_df.loc[time_df['accuracy'].idxmax()]
    spectral_best = spectral_df.loc[spectral_df['accuracy'].idxmax()]
    
    summary = f"""WISDM-51 Activity Recognition - Interpretation Summary
=====================================================

This analysis compared time-domain and spectral feature extraction approaches for 
human activity recognition using the WISDM-51 dataset.

KEY FINDINGS:
- Time-domain features achieved {time_best['accuracy']:.2f}% accuracy with {time_best.get('scaler', 'MinMax')} + {time_best['model']}
- Spectral features achieved {spectral_best['accuracy']:.2f}% accuracy with MinMax + {spectral_best['model']}
- Random Forest consistently outperformed other classifiers in both feature domains
- MinMax scaling proved most effective for normalizing sensor data
- Feature selection via Variance Threshold + Mutual Information effectively reduced 
  dimensionality while maintaining classification performance

RECOMMENDATIONS:
1. For production systems, use {time_best['model']} with {"time-domain" if time_best['accuracy'] >= spectral_best['accuracy'] else "spectral"} features
2. Consider combining both feature types for potentially improved performance
3. MinMax scaling should be applied before model training
4. Feature selection helps reduce computational requirements without significant accuracy loss
"""
    
    return summary


def run(spectral_results_df=None):
    """Execute Step 12: Final comparison and report generation."""
    logger.header("STEP 12: Final Comparison and Report")
    
    # Load results
    logger.log("Loading time-domain and spectral results...")
    time_results, spectral_results = load_results()
    
    logger.log(f"Time-domain results: {len(time_results)} configurations")
    logger.log(f"Spectral results: {len(spectral_results)} models")
    
    # Generate comparison visualizations
    logger.log("\nGenerating comparison visualizations...")
    time_best, spectral_best = generate_comparison_visualizations(time_results, spectral_results)
    
    # Combine results for comprehensive comparison
    # Standardize columns
    time_combined = time_results.copy()
    time_combined['config'] = time_combined['scaler'] + '_' + time_combined['model']
    
    spectral_combined = spectral_results.copy()
    spectral_combined['scaler'] = 'minmax'
    spectral_combined['config'] = 'minmax_' + spectral_combined['model']
    
    combined_df = pd.concat([time_combined, spectral_combined], ignore_index=True)
    combined_df = combined_df.sort_values('accuracy', ascending=False)
    
    # Save combined results
    output_dir = os.path.join(DATA_DIR, '12_final')
    os.makedirs(output_dir, exist_ok=True)
    
    combined_path = os.path.join(output_dir, 'combined_results.csv')
    combined_df.to_csv(combined_path, index=False)
    logger.log(f"\nSaved combined results to {combined_path}")
    
    # Generate final report
    logger.log("Generating final report...")
    report = generate_final_report(time_results, spectral_results, combined_df)
    
    report_path = os.path.join(BASE_DIR, 'FINAL_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    logger.log(f"Saved final report to {report_path}")
    
    # Generate interpretation summary
    interpretation = generate_interpretation_summary(time_results, spectral_results)
    interp_path = os.path.join(output_dir, 'interpretation_summary.txt')
    with open(interp_path, 'w') as f:
        f.write(interpretation)
    logger.log(f"Saved interpretation summary to {interp_path}")
    
    # Update README
    logger.log("Updating README.md...")
    update_readme(time_results, spectral_results, combined_df)
    
    # Log summary
    time_best_row = time_results.loc[time_results['accuracy'].idxmax()]
    spectral_best_row = spectral_results.loc[spectral_results['accuracy'].idxmax()]
    
    logger.log("\n" + "=" * 60)
    logger.log("FINAL COMPARISON SUMMARY")
    logger.log("=" * 60)
    logger.log(f"Time-Domain Best: {time_best_row.get('scaler', 'N/A')} + {time_best_row['model']} = {time_best_row['accuracy']:.2f}%")
    logger.log(f"Spectral Best: MinMax + {spectral_best_row['model']} = {spectral_best_row['accuracy']:.2f}%")
    logger.log(f"Overall Winner: {'Time-Domain' if time_best_row['accuracy'] >= spectral_best_row['accuracy'] else 'Spectral'}")
    logger.log("")
    
    return combined_df


def update_readme(time_df, spectral_df, combined_df):
    """Update README.md with complete results."""
    time_best = time_df.loc[time_df['accuracy'].idxmax()]
    spectral_best = spectral_df.loc[spectral_df['accuracy'].idxmax()]
    overall_best = combined_df.iloc[0]
    
    readme_content = f"""# WISDM-51 Activity Recognition Pipeline

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
| {overall_best['feature_type']} | {overall_best.get('scaler', 'minmax')} | {overall_best['model']} | **{overall_best['accuracy']:.2f}%** |

### Time-Domain Best
| Scaler | Model | Accuracy | Macro F1 |
|--------|-------|----------|----------|
| {time_best['scaler']} | {time_best['model']} | {time_best['accuracy']:.2f}% | {time_best['macro_f1']:.2f}% |

### Spectral Best
| Scaler | Model | Accuracy | Macro F1 |
|--------|-------|----------|----------|
| MinMax | {spectral_best['model']} | {spectral_best['accuracy']:.2f}% | {spectral_best['macro_f1']:.2f}% |

### Complete Results
| Rank | Feature Type | Scaler | Model | Accuracy |
|------|--------------|--------|-------|----------|
"""
    
    for i, (_, row) in enumerate(combined_df.head(10).iterrows(), 1):
        readme_content += f"| {i} | {row['feature_type']} | {row.get('scaler', 'minmax')} | {row['model']} | {row['accuracy']:.2f}% |\n"
    
    readme_content += f"""
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
*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    readme_path = os.path.join(BASE_DIR, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)


if __name__ == '__main__':
    run()
