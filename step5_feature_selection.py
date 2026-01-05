"""
WISDM-51 Activity Recognition Pipeline
Step 5: Feature Selection

Performs variance threshold filtering and mutual information-based selection.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from config import DATA_DIR, VIS_DIR, SCALER_NAMES, METADATA_COLS, RANDOM_STATE, NUM_FEATURES
from logger import logger


def perform_feature_selection(df, feature_cols):
    """
    Select features using variance threshold + mutual information.
    Returns: selected feature names, selection stats dict
    """
    X = df[feature_cols].values
    y = df['activity_label'].values
    
    # Variance threshold filter
    var_selector = VarianceThreshold(threshold=0.01)
    var_selector.fit(X)
    var_mask = var_selector.get_support()
    var_features = [feature_cols[i] for i, m in enumerate(var_mask) if m]
    
    logger.log(f"  Variance threshold: {len(var_features)}/{len(feature_cols)} features passed")
    
    if len(var_features) == 0:
        logger.log("  WARNING: No features passed variance threshold!")
        return feature_cols[:NUM_FEATURES], {'variance_passed': 0, 'mi_top_n': 0}
    
    # Compute mutual information on variance-filtered features
    X_var = df[var_features].values
    mi_scores = mutual_info_classif(X_var, y, random_state=RANDOM_STATE)
    
    # Sort by MI score and take top features
    mi_ranking = sorted(zip(var_features, mi_scores), key=lambda x: x[1], reverse=True)
    selected_features = [f for f, _ in mi_ranking[:NUM_FEATURES]]
    
    logger.log(f"  Selected top {len(selected_features)} features by mutual information")
    
    stats = {
        'variance_passed': len(var_features),
        'mi_top_n': len(selected_features),
        'mi_scores': dict(mi_ranking)
    }
    
    return selected_features, stats


def generate_selection_visualizations(mi_scores_dict):
    """Generate visualization of feature importance scores."""
    vis_dir = os.path.join(VIS_DIR, 'feature_selection')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Sort and plot top 30 features
    sorted_scores = sorted(mi_scores_dict.items(), key=lambda x: x[1], reverse=True)[:30]
    features = [f[0] for f in sorted_scores]
    scores = [f[1] for f in sorted_scores]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = sns.color_palette('viridis', len(features))
    ax.barh(range(len(features)), scores, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel('Mutual Information Score')
    ax.set_title('Top 30 Features by Mutual Information')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'mi_feature_importance.png'), dpi=300)
    plt.close()


def run(scaled_dfs=None):
    """Execute Step 5: Feature selection for each scaled dataset."""
    logger.header("STEP 5: Feature Selection")
    
    output_dir = os.path.join(DATA_DIR, '05_selected')
    os.makedirs(output_dir, exist_ok=True)
    
    selected_dfs = {}
    all_selected_features = {}
    
    for scaler_name in SCALER_NAMES:
        logger.log(f"Processing {scaler_name} scaled data...")
        
        if scaled_dfs and scaler_name in scaled_dfs:
            df = scaled_dfs[scaler_name]
        else:
            df = pd.read_csv(os.path.join(DATA_DIR, '04_scaled', f'{scaler_name}_scaled.csv'))
        
        feature_cols = [c for c in df.columns if c not in METADATA_COLS]
        
        selected_features, stats = perform_feature_selection(df, feature_cols)
        all_selected_features[scaler_name] = selected_features
        
        # Generate visualization for first scaler
        if scaler_name == SCALER_NAMES[0] and 'mi_scores' in stats:
            generate_selection_visualizations(stats['mi_scores'])
        
        # Create selected dataset
        selected_df = df[METADATA_COLS + selected_features].copy()
        
        output_path = os.path.join(output_dir, f'{scaler_name}_selected.csv')
        selected_df.to_csv(output_path, index=False)
        selected_dfs[scaler_name] = selected_df
        
        logger.log(f"  Saved {scaler_name}_selected.csv with {len(selected_features)} features")
    
    # Save selected feature names
    with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
        for scaler_name, features in all_selected_features.items():
            f.write(f"=== {scaler_name.upper()} ===\n")
            for feat in features:
                f.write(f"{feat}\n")
            f.write("\n")
    
    logger.log("")
    return selected_dfs


if __name__ == '__main__':
    run()
