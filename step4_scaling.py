"""
WISDM-51 Activity Recognition Pipeline
Step 4: Data Scaling

Applies MinMax, Standard, and Robust scaling to extracted features.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from config import DATA_DIR, VIS_DIR, SCALER_NAMES, METADATA_COLS
from logger import logger


def get_scalers():
    """Return dictionary of scaler instances."""
    return {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
        'robust': RobustScaler()
    }


def generate_scaling_visualizations(raw_df, scaled_df, feature_cols, scaler_name):
    """Generate boxplot and histogram comparing raw vs scaled features."""
    vis_dir = os.path.join(VIS_DIR, 'scaling_comparison')
    os.makedirs(vis_dir, exist_ok=True)
    
    selected_features = feature_cols[:15]
    
    # Boxplot
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(selected_features):
        if i < len(axes):
            data = [raw_df[col].dropna(), scaled_df[col].dropna()]
            axes[i].boxplot(data, labels=['Raw', 'Scaled'])
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(labelsize=7)
    
    plt.suptitle(f'{scaler_name.upper()} Scaling - Before/After Comparison (Boxplot)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{scaler_name}_boxplot.png'), dpi=300)
    plt.close()
    
    # Histogram
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(selected_features):
        if i < len(axes):
            axes[i].hist(raw_df[col].dropna(), bins=30, alpha=0.5, label='Raw', color='blue')
            axes[i].hist(scaled_df[col].dropna(), bins=30, alpha=0.5, label='Scaled', color='orange')
            axes[i].set_title(col, fontsize=9)
            axes[i].legend(fontsize=6)
            axes[i].tick_params(labelsize=7)
    
    plt.suptitle(f'{scaler_name.upper()} Scaling - Before/After Comparison (Histogram)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{scaler_name}_histogram.png'), dpi=300)
    plt.close()


def run(features_df=None):
    """Execute Step 4: Apply different scaling methods."""
    logger.header("STEP 4: Data Scaling")
    
    if features_df is None:
        features_df = pd.read_csv(os.path.join(DATA_DIR, '03_features', 'features_raw.csv'))
    
    feature_cols = [c for c in features_df.columns if c not in METADATA_COLS]
    scalers = get_scalers()
    
    scaled_dfs = {}
    output_dir = os.path.join(DATA_DIR, '04_scaled')
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute raw stats
    raw_stats = features_df[feature_cols].describe().loc[['mean', 'std', 'min', 'max']].T
    raw_stats.columns = ['raw_mean', 'raw_std', 'raw_min', 'raw_max']
    comparison_df = raw_stats.copy()
    
    for scaler_name in SCALER_NAMES:
        logger.log(f"Applying {scaler_name} scaling...")
        scaler = scalers[scaler_name]
        
        # Scale features
        scaled_features = scaler.fit_transform(features_df[feature_cols])
        scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
        
        # Add metadata
        for col in METADATA_COLS:
            scaled_df[col] = features_df[col].values
        
        # Save
        output_path = os.path.join(output_dir, f'{scaler_name}_scaled.csv')
        scaled_df.to_csv(output_path, index=False)
        scaled_dfs[scaler_name] = scaled_df
        
        # Add to comparison stats
        scaled_stats = scaled_df[feature_cols].describe().loc[['mean', 'std', 'min', 'max']].T
        scaled_stats.columns = [f'{scaler_name}_mean', f'{scaler_name}_std', f'{scaler_name}_min', f'{scaler_name}_max']
        comparison_df = pd.concat([comparison_df, scaled_stats], axis=1)
        
        # Generate visualizations
        generate_scaling_visualizations(features_df, scaled_df, feature_cols, scaler_name)
        
        logger.log(f"  Saved {scaler_name}_scaled.csv")
    
    # Save comparison stats
    comparison_df.to_csv(os.path.join(output_dir, 'scaling_comparison_stats.csv'))
    logger.log("Saved scaling comparison statistics")
    logger.log("")
    
    return scaled_dfs


if __name__ == '__main__':
    run()
