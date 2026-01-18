"""
WISDM-51 Activity Recognition Pipeline
Step 10: Spectral Feature Selection

Performs variance threshold filtering and mutual information-based selection
on spectral features.

Justification:
Variance Threshold + Mutual Information was used successfully in time-domain 
feature selection. This two-stage approach first removes low-variance features 
(noise/constants), then ranks remaining features by their mutual information 
with the target class, selecting the most discriminative features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from config import DATA_DIR, VIS_DIR, METADATA_COLS, RANDOM_STATE, NUM_SPECTRAL_FEATURES
from logger import logger


def perform_feature_selection(df, feature_cols):
    """
    Select features using variance threshold + mutual information.
    
    Returns:
    --------
    tuple: (selected_features, stats_dict)
    """
    X = df[feature_cols].values
    y = df['activity_label'].values
    
    # Stage 1: Variance threshold filter
    logger.log("Stage 1: Variance Threshold Filtering...")
    var_selector = VarianceThreshold(threshold=0.01)
    var_selector.fit(X)
    var_mask = var_selector.get_support()
    var_features = [feature_cols[i] for i, m in enumerate(var_mask) if m]
    
    logger.log(f"  {len(var_features)}/{len(feature_cols)} features passed variance threshold")
    
    if len(var_features) == 0:
        logger.log("  WARNING: No features passed variance threshold! Using all features.")
        var_features = feature_cols
    
    # Stage 2: Mutual Information ranking
    logger.log("Stage 2: Mutual Information Ranking...")
    X_var = df[var_features].values
    mi_scores = mutual_info_classif(X_var, y, random_state=RANDOM_STATE)
    
    # Sort by MI score and take top features
    mi_ranking = sorted(zip(var_features, mi_scores), key=lambda x: x[1], reverse=True)
    
    # Select top N features (adjust based on available features)
    num_to_select = min(NUM_SPECTRAL_FEATURES, len(mi_ranking))
    selected_features = [f for f, _ in mi_ranking[:num_to_select]]
    
    logger.log(f"  Selected top {len(selected_features)} features by mutual information")
    
    stats = {
        'total_features': len(feature_cols),
        'variance_passed': len(var_features),
        'selected': len(selected_features),
        'mi_scores': dict(mi_ranking)
    }
    
    return selected_features, stats


def generate_selection_visualizations(mi_scores_dict, selected_features):
    """Generate visualization of feature importance scores."""
    vis_dir = os.path.join(VIS_DIR, 'spectral_selection')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Sort and get top 30 features for visualization
    sorted_scores = sorted(mi_scores_dict.items(), key=lambda x: x[1], reverse=True)[:30]
    features = [f[0] for f in sorted_scores]
    scores = [f[1] for f in sorted_scores]
    
    # Highlight selected features
    colors = ['steelblue' if f in selected_features else 'lightgray' for f in features]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(range(len(features)), scores, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel('Mutual Information Score')
    ax.set_title('Top 30 Spectral Features by Mutual Information\n(Blue = Selected, Gray = Not Selected)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'spectral_mi_feature_importance.png'), dpi=300)
    plt.close()
    
    logger.log("Saved spectral feature importance visualization")


def run(scaled_df=None):
    """Execute Step 10: Feature selection on spectral features."""
    logger.header("STEP 10: Spectral Feature Selection")
    
    # Load scaled spectral features if not provided
    if scaled_df is None:
        input_path = os.path.join(DATA_DIR, '09_spectral_scaled', 'SCALED_SPECTRAL_FEATURES.csv')
        logger.log(f"Loading scaled features from {input_path}")
        scaled_df = pd.read_csv(input_path)
    
    logger.log(f"Processing {len(scaled_df):,} samples...")
    
    # Get feature columns
    feature_cols = [c for c in scaled_df.columns if c not in METADATA_COLS]
    logger.log(f"Total features before selection: {len(feature_cols)}")
    
    # Perform feature selection
    selected_features, stats = perform_feature_selection(scaled_df, feature_cols)
    
    # Generate visualizations
    generate_selection_visualizations(stats['mi_scores'], selected_features)
    
    # Create selected features DataFrame
    selected_df = scaled_df[METADATA_COLS + selected_features].copy()
    
    # Save outputs
    output_dir = os.path.join(DATA_DIR, '10_spectral_selected')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save selected features CSV
    selected_path = os.path.join(output_dir, 'FINAL_SELECTED_SPECTRAL_FEATURES.csv')
    selected_df.to_csv(selected_path, index=False)
    logger.log(f"Saved selected features to {selected_path}")
    
    # Save selected feature list
    features_list_path = os.path.join(output_dir, 'selected_spectral_features.txt')
    with open(features_list_path, 'w') as f:
        f.write("WISDM-51 Selected Spectral Features\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Selection Method: Variance Threshold + Mutual Information\n")
        f.write(f"Total Features: {stats['total_features']}\n")
        f.write(f"Variance Threshold Passed: {stats['variance_passed']}\n")
        f.write(f"Final Selected: {stats['selected']}\n\n")
        f.write("Selected Features (ranked by MI score):\n")
        f.write("-" * 50 + "\n")
        
        for rank, feat in enumerate(selected_features, 1):
            mi_score = stats['mi_scores'].get(feat, 0)
            f.write(f"{rank:2}. {feat:40} (MI: {mi_score:.4f})\n")
    
    logger.log(f"Saved feature list to {features_list_path}")
    
    # Log summary
    logger.log(f"\nSelection Summary:")
    logger.log(f"  Original features: {stats['total_features']}")
    logger.log(f"  After variance filter: {stats['variance_passed']}")
    logger.log(f"  Final selected: {stats['selected']}")
    logger.log(f"  Final shape: {selected_df.shape}")
    logger.log("")
    
    return selected_df


if __name__ == '__main__':
    run()
