"""
WISDM-51 Activity Recognition Pipeline
Step 9: Spectral Feature Scaling

Applies MinMax scaling to spectral features.

Justification:
MinMax scaling was chosen because it achieved the highest accuracy (72.21%) 
with RandomForest in time-domain features. MinMax preserves the original 
distribution shape while ensuring all features are in [0,1] range, which 
is beneficial for distance-based algorithms and prevents features with 
larger magnitudes from dominating.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from config import DATA_DIR, VIS_DIR, METADATA_COLS
from logger import logger


def generate_scaling_visualizations(raw_df, scaled_df, feature_cols):
    """Generate before/after scaling comparison visualizations."""
    vis_dir = os.path.join(VIS_DIR, 'spectral_scaling')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Select first 15 features for visualization
    selected_features = feature_cols[:15]
    
    # Boxplot comparison
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(selected_features):
        if i < len(axes):
            data = [raw_df[col].dropna(), scaled_df[col].dropna()]
            axes[i].boxplot(data, labels=['Raw', 'Scaled'])
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(labelsize=7)
    
    plt.suptitle('MinMax Scaling - Before/After Comparison (Boxplot)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'spectral_minmax_boxplot.png'), dpi=300)
    plt.close()
    
    # Histogram comparison
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(selected_features):
        if i < len(axes):
            axes[i].hist(raw_df[col].dropna(), bins=30, alpha=0.5, label='Raw', color='blue')
            axes[i].hist(scaled_df[col].dropna(), bins=30, alpha=0.5, label='Scaled', color='orange')
            axes[i].set_title(col, fontsize=9)
            axes[i].legend(fontsize=6)
            axes[i].tick_params(labelsize=7)
    
    plt.suptitle('MinMax Scaling - Before/After Comparison (Histogram)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'spectral_minmax_histogram.png'), dpi=300)
    plt.close()
    
    logger.log("Saved spectral scaling visualizations")


def run(spectral_df=None):
    """Execute Step 9: Apply MinMax scaling to spectral features."""
    logger.header("STEP 9: Spectral Feature Scaling")
    
    # Load spectral features if not provided
    if spectral_df is None:
        input_path = os.path.join(DATA_DIR, '08_spectral', 'SPECTRAL_FEATURES.csv')
        logger.log(f"Loading spectral features from {input_path}")
        spectral_df = pd.read_csv(input_path)
    
    logger.log(f"Scaling {len(spectral_df):,} samples...")
    
    # Get feature columns
    feature_cols = [c for c in spectral_df.columns if c not in METADATA_COLS]
    logger.log(f"Number of features to scale: {len(feature_cols)}")
    
    # Compute raw statistics
    raw_stats = spectral_df[feature_cols].describe().loc[['mean', 'std', 'min', 'max']].T
    raw_stats.columns = ['raw_mean', 'raw_std', 'raw_min', 'raw_max']
    
    # Apply MinMax scaling
    logger.log("Applying MinMax scaling...")
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(spectral_df[feature_cols])
    
    # Create scaled DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
    
    # Add metadata columns
    for col in METADATA_COLS:
        if col in spectral_df.columns:
            scaled_df[col] = spectral_df[col].values
    
    # Compute scaled statistics
    scaled_stats = scaled_df[feature_cols].describe().loc[['mean', 'std', 'min', 'max']].T
    scaled_stats.columns = ['scaled_mean', 'scaled_std', 'scaled_min', 'scaled_max']
    
    # Combine statistics
    comparison_stats = pd.concat([raw_stats, scaled_stats], axis=1)
    
    # Save scaled features
    output_dir = os.path.join(DATA_DIR, '09_spectral_scaled')
    os.makedirs(output_dir, exist_ok=True)
    
    scaled_path = os.path.join(output_dir, 'SCALED_SPECTRAL_FEATURES.csv')
    scaled_df.to_csv(scaled_path, index=False)
    logger.log(f"Saved scaled features to {scaled_path}")
    
    # Save comparison statistics
    stats_path = os.path.join(output_dir, 'scaling_spectral_comparison_stats.csv')
    comparison_stats.to_csv(stats_path)
    logger.log(f"Saved comparison stats to {stats_path}")
    
    # Generate visualizations
    generate_scaling_visualizations(spectral_df, scaled_df, feature_cols)
    
    # Log summary
    logger.log(f"\nScaling Summary:")
    logger.log(f"  Features scaled: {len(feature_cols)}")
    logger.log(f"  Raw range: [{raw_stats['raw_min'].min():.4f}, {raw_stats['raw_max'].max():.4f}]")
    logger.log(f"  Scaled range: [{scaled_stats['scaled_min'].min():.4f}, {scaled_stats['scaled_max'].max():.4f}]")
    logger.log(f"  Final shape: {scaled_df.shape}")
    logger.log("")
    
    return scaled_df


if __name__ == '__main__':
    run()
