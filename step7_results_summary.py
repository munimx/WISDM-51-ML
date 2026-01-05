"""
WISDM-51 Activity Recognition Pipeline
Step 7: Results Summary and README Update

Generates final summary and updates project README with results.
"""

import os
import pandas as pd
from datetime import datetime

from config import BASE_DIR, DATA_DIR, SCALER_NAMES
from logger import logger


def generate_summary(results_df=None):
    """Generate comprehensive results summary."""
    if results_df is None:
        results_df = pd.read_csv(os.path.join(DATA_DIR, '06_results', 'model_results.csv'))
    
    summary_lines = []
    summary_lines.append("="*70)
    summary_lines.append("WISDM-51 ACTIVITY RECOGNITION - RESULTS SUMMARY")
    summary_lines.append("="*70)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    # Best overall
    best = results_df.iloc[0]
    summary_lines.append("BEST MODEL COMBINATION:")
    summary_lines.append(f"  Scaler: {best['scaler']}")
    summary_lines.append(f"  Model: {best['model']}")
    summary_lines.append(f"  Accuracy: {best['accuracy']:.2f}%")
    summary_lines.append(f"  Macro F1: {best['macro_f1']:.2f}%")
    summary_lines.append("")
    
    # By scaler
    summary_lines.append("BEST MODEL BY SCALER:")
    for scaler in SCALER_NAMES:
        scaler_df = results_df[results_df['scaler'] == scaler]
        if not scaler_df.empty:
            best_row = scaler_df.iloc[0]
            summary_lines.append(f"  {scaler:10}: {best_row['model']:12} ({best_row['accuracy']:.2f}%)")
    summary_lines.append("")
    
    # By model
    summary_lines.append("BEST SCALER BY MODEL:")
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model].sort_values('accuracy', ascending=False)
        if not model_df.empty:
            best_row = model_df.iloc[0]
            summary_lines.append(f"  {model:12}: {best_row['scaler']:10} ({best_row['accuracy']:.2f}%)")
    summary_lines.append("")
    
    # Full ranking
    summary_lines.append("FULL RANKING:")
    summary_lines.append("-"*50)
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        summary_lines.append(f"  {i:2}. {row['scaler']:10} + {row['model']:12} = {row['accuracy']:.2f}%")
    summary_lines.append("")
    
    return "\n".join(summary_lines)


def update_readme(results_df=None):
    """Update README.md with pipeline results."""
    if results_df is None:
        results_df = pd.read_csv(os.path.join(DATA_DIR, '06_results', 'model_results.csv'))
    
    best = results_df.iloc[0]
    
    readme_content = f"""# WISDM-51 Activity Recognition Pipeline

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
| {best['scaler']} | {best['model']} | {best['accuracy']:.2f}% | {best['macro_f1']:.2f}% |

### All Results (Sorted by Accuracy)
| Rank | Scaler | Model | Accuracy |
|------|--------|-------|----------|
"""
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        readme_content += f"| {i} | {row['scaler']} | {row['model']} | {row['accuracy']:.2f}% |\n"
    
    readme_content += f"""
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
*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return readme_content


def run(results_df=None):
    """Execute Step 7: Generate summary and update README."""
    logger.header("STEP 7: Results Summary")
    
    if results_df is None:
        results_df = pd.read_csv(os.path.join(DATA_DIR, '06_results', 'model_results.csv'))
    
    # Generate and save summary
    summary = generate_summary(results_df)
    
    summary_path = os.path.join(DATA_DIR, '06_results', 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    logger.log("Generated results summary")
    print(summary)
    
    # Update README
    readme_content = update_readme(results_df)
    readme_path = os.path.join(BASE_DIR, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.log("Updated README.md")
    logger.log("")
    
    return summary


if __name__ == '__main__':
    run()
