"""
WISDM-51 Activity Recognition Pipeline
Step 3c: Combine All Feature Sets - OPTIMIZED VERSION

Merges basic time-domain, advanced, and spectral features into 
a comprehensive feature set for improved classification.
Uses efficient pandas operations for fast merging.
"""

import os
import pandas as pd
import numpy as np

from config import DATA_DIR, METADATA_COLS
from logger import logger


def run(basic_df=None, advanced_df=None, spectral_df=None):
    """Combine all feature sets into a single comprehensive dataset - OPTIMIZED."""
    logger.header("STEP 3C: Combining Feature Sets (Optimized)")
    
    # Load all feature sets if not provided
    logger.log("Loading feature sets...")
    
    if basic_df is None:
        basic_path = os.path.join(DATA_DIR, '03_features', 'features_raw.csv')
        basic_df = pd.read_csv(basic_path)
    
    if advanced_df is None:
        advanced_path = os.path.join(DATA_DIR, '03b_advanced_features', 'advanced_features.csv')
        advanced_df = pd.read_csv(advanced_path)
    
    if spectral_df is None:
        spectral_path = os.path.join(DATA_DIR, '08_spectral', 'SPECTRAL_FEATURES.csv')
        spectral_df = pd.read_csv(spectral_path)
    
    logger.log(f"  Basic features: {basic_df.shape}")
    logger.log(f"  Advanced features: {advanced_df.shape}")
    logger.log(f"  Spectral features: {spectral_df.shape}")
    
    # Get feature columns only (exclude metadata)
    basic_feats = [c for c in basic_df.columns if c not in METADATA_COLS]
    advanced_feats = [c for c in advanced_df.columns if c not in METADATA_COLS]
    spectral_feats = [c for c in spectral_df.columns if c not in METADATA_COLS]
    
    # Check for and handle duplicate column names between feature sets
    basic_set = set(basic_feats)
    advanced_set = set(advanced_feats)
    spectral_set = set(spectral_feats)
    
    # Find overlapping columns
    adv_spec_overlap = advanced_set & spectral_set
    basic_adv_overlap = basic_set & advanced_set
    basic_spec_overlap = basic_set & spectral_set
    
    if adv_spec_overlap:
        logger.log(f"  Renaming {len(adv_spec_overlap)} duplicate columns in spectral (already in advanced)")
        spectral_rename = {c: f'sp_{c}' for c in adv_spec_overlap}
        spectral_df = spectral_df.rename(columns=spectral_rename)
        spectral_feats = [spectral_rename.get(c, c) for c in spectral_feats]
    
    if basic_adv_overlap:
        logger.log(f"  Renaming {len(basic_adv_overlap)} duplicate columns in advanced (already in basic)")
        adv_rename = {c: f'adv_{c}' for c in basic_adv_overlap}
        advanced_df = advanced_df.rename(columns=adv_rename)
        advanced_feats = [adv_rename.get(c, c) for c in advanced_feats]
    
    if basic_spec_overlap:
        logger.log(f"  Renaming {len(basic_spec_overlap)} duplicate columns in spectral (already in basic)")
        spec_rename = {c: f'sp_{c}' for c in basic_spec_overlap}
        spectral_df = spectral_df.rename(columns=spec_rename)
        spectral_feats = [spec_rename.get(c, c) for c in spectral_feats]
    
    logger.log(f"\nFeature counts:")
    logger.log(f"  Basic: {len(basic_feats)}")
    logger.log(f"  Advanced: {len(advanced_feats)}")
    logger.log(f"  Spectral: {len(spectral_feats)}")
    logger.log(f"  Total: {len(basic_feats) + len(advanced_feats) + len(spectral_feats)}")
    
    # Efficient concatenation using numpy arrays
    logger.log("\nCombining features efficiently...")
    
    # Keep metadata from basic_df
    metadata_cols = [c for c in METADATA_COLS if c in basic_df.columns]
    
    # Use numpy for faster concatenation
    meta_data = basic_df[metadata_cols].values
    basic_data = basic_df[basic_feats].values
    advanced_data = advanced_df[advanced_feats].values
    spectral_data = spectral_df[spectral_feats].values
    
    # Concatenate all arrays horizontally
    combined_data = np.hstack([meta_data, basic_data, advanced_data, spectral_data])
    
    # Create DataFrame with all column names
    all_columns = metadata_cols + basic_feats + advanced_feats + spectral_feats
    combined_df = pd.DataFrame(combined_data, columns=all_columns)
    
    # Convert metadata columns back to proper types
    if 'subject_id' in combined_df.columns:
        combined_df['subject_id'] = combined_df['subject_id'].astype(int)
    if 'activity_label' in combined_df.columns:
        combined_df['activity_label'] = combined_df['activity_label'].astype(int)
    
    # Handle any NaN/inf values in combined features (vectorized)
    all_feature_cols = basic_feats + advanced_feats + spectral_feats
    
    # Convert feature columns to numeric (handles any object dtypes from hstack)
    for col in all_feature_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    feature_data = combined_df[all_feature_cols].values.astype(np.float64)
    feature_data = np.where(np.isinf(feature_data), np.nan, feature_data)
    feature_data = np.nan_to_num(feature_data, 0)
    combined_df[all_feature_cols] = feature_data
    
    logger.log(f"\nCombined shape: {combined_df.shape}")
    
    # Save combined features
    output_dir = os.path.join(DATA_DIR, '03c_combined')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_features.csv')
    combined_df.to_csv(output_path, index=False)
    
    logger.log(f"Saved to {output_path}")
    
    # Save feature list
    features_list_path = os.path.join(output_dir, 'combined_feature_list.txt')
    with open(features_list_path, 'w') as f:
        f.write("WISDM-51 Combined Feature List\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total features: {len(all_feature_cols)}\n\n")
        
        f.write("=== BASIC TIME-DOMAIN FEATURES ===\n")
        for feat in basic_feats:
            f.write(f"  {feat}\n")
        
        f.write("\n=== ADVANCED FEATURES ===\n")
        for feat in advanced_feats:
            f.write(f"  {feat}\n")
        
        f.write("\n=== SPECTRAL FEATURES ===\n")
        for feat in spectral_feats:
            f.write(f"  {feat}\n")
    
    logger.log(f"Saved feature list to {features_list_path}")
    logger.log("")
    
    return combined_df


if __name__ == '__main__':
    run()
