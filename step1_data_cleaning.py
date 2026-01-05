"""
WISDM-51 Activity Recognition Pipeline
Step 1: Data Loading and Cleaning

Loads raw sensor data from txt files, cleans it, and saves consolidated CSV.
"""

import os
import numpy as np
import pandas as pd

from config import (
    RAW_DIR, DATA_DIR, ACTIVITY_MAPPING
)
from logger import logger


def load_raw_data():
    """Load all raw sensor data from the raw folder."""
    all_data = []
    
    data_paths = [
        ('phone', 'accel', os.path.join(RAW_DIR, 'phone', 'accel')),
        ('phone', 'gyro', os.path.join(RAW_DIR, 'phone', 'gyro')),
        ('watch', 'accel', os.path.join(RAW_DIR, 'watch', 'accel')),
        ('watch', 'gyro', os.path.join(RAW_DIR, 'watch', 'gyro')),
    ]
    
    for device, sensor_type, path in data_paths:
        if not os.path.exists(path):
            logger.log(f"WARNING: Path not found: {path}")
            continue
        
        files = [f for f in os.listdir(path) if f.endswith('.txt') and not f.startswith('.')]
        logger.log(f"Loading {len(files)} files from {device}/{sensor_type}")
        
        type_data = []
        for filename in sorted(files):
            filepath = os.path.join(path, filename)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip().rstrip(';').strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) != 6:
                        continue
                    
                    try:
                        type_data.append({
                            'subject_id': int(parts[0]),
                            'activity_code': parts[1].strip(),
                            'timestamp': int(parts[2]),
                            'x': float(parts[3]),
                            'y': float(parts[4]),
                            'z': float(parts[5]),
                            'sensor_type': sensor_type,
                            'device': device
                        })
                    except (ValueError, IndexError):
                        continue
                        
            except Exception as e:
                logger.log(f"Error reading {filename}: {e}")
        
        logger.log(f"  Loaded {len(type_data):,} rows from {device}/{sensor_type}")
        all_data.extend(type_data)
    
    df = pd.DataFrame(all_data)
    logger.log(f"Total rows loaded: {len(df):,}")
    
    return df


def clean_data(df):
    """Clean the loaded data by removing invalid entries."""
    initial_rows = len(df)
    
    # Remove NaN values
    nan_mask = df[['x', 'y', 'z']].isna().any(axis=1)
    nan_count = nan_mask.sum()
    df = df[~nan_mask].copy()
    
    # Remove infinite values
    inf_mask = np.isinf(df[['x', 'y', 'z']]).any(axis=1)
    inf_count = inf_mask.sum()
    df = df[~inf_mask].copy()
    
    # Remove invalid activity codes
    valid_codes = set(ACTIVITY_MAPPING.keys())
    invalid_mask = ~df['activity_code'].isin(valid_codes)
    invalid_count = invalid_mask.sum()
    df = df[~invalid_mask].copy()
    
    logger.log(f"Rows removed - NaN: {nan_count}, Inf: {inf_count}, Invalid activity: {invalid_count}")
    logger.log(f"Rows after cleaning: {len(df):,}")
    
    # Add activity_label column
    df['activity_label'] = df['activity_code'].map(ACTIVITY_MAPPING)
    
    # Sort data
    df = df.sort_values(['device', 'sensor_type', 'subject_id', 'activity_code', 'timestamp'])
    df = df.reset_index(drop=True)
    
    # Validate sampling rate
    logger.log("Validating sampling rate...")
    sample_df = df[(df['device'] == 'phone') & (df['sensor_type'] == 'accel') & 
                   (df['subject_id'] == 1600) & (df['activity_code'] == 'A')].head(100)
    if len(sample_df) > 1:
        time_diffs = sample_df['timestamp'].diff().dropna() / 1e6
        mean_diff = time_diffs.mean()
        logger.log(f"Sample mean time diff: {mean_diff:.2f} ms (expected ~50 ms)")
    
    # Reorder columns
    df = df[['subject_id', 'activity_code', 'activity_label', 'timestamp', 'x', 'y', 'z', 'sensor_type', 'device']]
    
    return df


def run():
    """Execute Step 1: Load and clean data."""
    logger.header("STEP 1: Data Loading and Cleaning")
    
    # Load data
    df = load_raw_data()
    
    # Clean data
    df = clean_data(df)
    
    # Save cleaned data
    output_dir = os.path.join(DATA_DIR, '01_cleaned')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cleaned_data.csv')
    df.to_csv(output_path, index=False)
    
    logger.log(f"Saved cleaned data to {output_path}")
    logger.log(f"Final shape: {df.shape}")
    logger.log("")
    
    return df


if __name__ == '__main__':
    run()
