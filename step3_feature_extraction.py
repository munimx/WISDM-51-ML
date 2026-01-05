"""
WISDM-51 Activity Recognition Pipeline
Step 3: Feature Extraction

Extracts 60 time-domain features from windowed data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks

from config import DATA_DIR, VIS_DIR, WINDOW_SIZE, METADATA_COLS
from logger import logger


def compute_features(x, y, z):
    """Compute all 20 features for each axis."""
    features = {}
    
    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        axis_data = np.array(axis_data)
        
        # Basic statistics
        features[f'mean_{axis_name}'] = np.mean(axis_data)
        features[f'median_{axis_name}'] = np.median(axis_data)
        std_val = np.std(axis_data)
        features[f'std_{axis_name}'] = std_val
        features[f'variance_{axis_name}'] = np.var(axis_data)
        features[f'min_{axis_name}'] = np.min(axis_data)
        features[f'max_{axis_name}'] = np.max(axis_data)
        features[f'range_{axis_name}'] = np.max(axis_data) - np.min(axis_data)
        
        # Distribution statistics
        features[f'skewness_{axis_name}'] = stats.skew(axis_data)
        features[f'kurtosis_{axis_name}'] = stats.kurtosis(axis_data)
        q75, q25 = np.percentile(axis_data, [75, 25])
        features[f'iqr_{axis_name}'] = q75 - q25
        features[f'mad_{axis_name}'] = np.mean(np.abs(axis_data - np.mean(axis_data)))
        
        # Signal characteristics
        features[f'rms_{axis_name}'] = np.sqrt(np.mean(axis_data ** 2))
        features[f'zcr_{axis_name}'] = np.sum(np.diff(np.sign(axis_data)) != 0)
        
        # Autocorrelation
        if len(axis_data) > 1:
            autocorr = np.corrcoef(axis_data[:-1], axis_data[1:])[0, 1]
            features[f'autocorr_{axis_name}'] = autocorr if not np.isnan(autocorr) else 0
        else:
            features[f'autocorr_{axis_name}'] = 0
        
        # Energy features
        features[f'sma_{axis_name}'] = np.mean(np.abs(axis_data))
        features[f'energy_{axis_name}'] = np.mean(axis_data ** 2)
        
        # Hjorth parameters
        hjorth_activity = np.var(axis_data)
        features[f'hjorth_activity_{axis_name}'] = hjorth_activity
        
        first_deriv = np.diff(axis_data)
        var_deriv = np.var(first_deriv) if len(first_deriv) > 0 else 0
        hjorth_mobility = np.sqrt(var_deriv / hjorth_activity) if hjorth_activity > 0 else 0
        features[f'hjorth_mobility_{axis_name}'] = hjorth_mobility
        
        if len(first_deriv) > 1 and hjorth_mobility > 0:
            second_deriv = np.diff(first_deriv)
            var_second_deriv = np.var(second_deriv) if len(second_deriv) > 0 else 0
            mobility_deriv = np.sqrt(var_second_deriv / var_deriv) if var_deriv > 0 else 0
            hjorth_complexity = mobility_deriv / hjorth_mobility if hjorth_mobility > 0 else 0
        else:
            hjorth_complexity = 0
        features[f'hjorth_complexity_{axis_name}'] = hjorth_complexity
        
        # Peak count
        prominence_threshold = 0.5 * std_val if std_val > 0 else 0.1
        peaks, _ = find_peaks(axis_data, prominence=prominence_threshold)
        features[f'peak_count_{axis_name}'] = len(peaks)
    
    return features


def generate_visualizations(df, feature_cols):
    """Generate histogram and boxplot for raw features."""
    vis_dir = os.path.join(VIS_DIR, 'feature_distributions')
    os.makedirs(vis_dir, exist_ok=True)
    
    n_features = min(60, len(feature_cols))
    selected_features = feature_cols[:n_features]
    
    # Histogram
    fig, axes = plt.subplots(10, 6, figsize=(20, 25))
    axes = axes.flatten()
    
    for i, col in enumerate(selected_features):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(col, fontsize=8)
            axes[i].tick_params(labelsize=6)
    
    for i in range(len(selected_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Raw Features Distribution (Histogram)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'raw_features_histogram.png'), dpi=300)
    plt.close()
    
    # Boxplot
    fig, axes = plt.subplots(10, 6, figsize=(20, 25))
    axes = axes.flatten()
    
    for i, col in enumerate(selected_features):
        if i < len(axes):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(col, fontsize=8)
            axes[i].tick_params(labelsize=6)
    
    for i in range(len(selected_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Raw Features Distribution (Boxplot)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'raw_features_boxplot.png'), dpi=300)
    plt.close()
    
    logger.log("Saved feature distribution visualizations")


def run(windows_df=None):
    """Execute Step 3: Extract features from windows."""
    logger.header("STEP 3: Feature Extraction")
    
    if windows_df is None:
        windows_df = pd.read_csv(os.path.join(DATA_DIR, '02_windowed', 'windowed_data.csv'))
    
    logger.log(f"Processing {len(windows_df):,} windows...")
    
    features_list = []
    for idx in range(len(windows_df)):
        row = windows_df.iloc[idx]
        
        x = [row[f'x_{i}'] for i in range(WINDOW_SIZE)]
        y = [row[f'y_{i}'] for i in range(WINDOW_SIZE)]
        z = [row[f'z_{i}'] for i in range(WINDOW_SIZE)]
        
        features = compute_features(x, y, z)
        features['subject_id'] = row['subject_id']
        features['activity_label'] = row['activity_label']
        features['sensor_type'] = row['sensor_type']
        features['device'] = row['device']
        
        features_list.append(features)
        
        if (idx + 1) % 10000 == 0:
            logger.log(f"  Processed {idx + 1:,} windows...")
    
    features_df = pd.DataFrame(features_list)
    
    # Handle NaN/inf
    feature_cols = [c for c in features_df.columns if c not in METADATA_COLS]
    features_df[feature_cols] = features_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    features_df[feature_cols] = features_df[feature_cols].fillna(0)
    
    # Save features
    output_dir = os.path.join(DATA_DIR, '03_features')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'features_raw.csv')
    features_df.to_csv(output_path, index=False)
    
    logger.log(f"Saved features to {output_path}")
    logger.log(f"Final shape: {features_df.shape}")
    
    # Save feature descriptions
    with open(os.path.join(output_dir, 'feature_descriptions.txt'), 'w') as f:
        f.write("WISDM-51 Feature Descriptions\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total features: {len(feature_cols)}\n\n")
        for i, col in enumerate(sorted(feature_cols), 1):
            f.write(f"{i}. {col}\n")
    
    # Generate visualizations
    logger.log("Generating feature distribution visualizations...")
    generate_visualizations(features_df, feature_cols)
    logger.log("")
    
    return features_df


if __name__ == '__main__':
    run()
