"""
WISDM-51 Activity Recognition Pipeline
Step 2: Windowing

Creates sliding windows from cleaned sensor data.
"""

import os
import numpy as np
import pandas as pd

from config import (
    DATA_DIR, ACTIVITY_MAPPING, WINDOW_SIZE, HOP_SIZE, ACTIVITY_NAMES
)
from logger import logger


def create_windows(df):
    """Create sliding windows from the cleaned data."""
    windows = []
    windows_padded = 0
    
    # Group by subject, activity, sensor, device
    groups = df.groupby(['subject_id', 'activity_code', 'sensor_type', 'device'])
    total_groups = len(groups)
    logger.log(f"Processing {total_groups} groups...")
    
    for (subject_id, activity_code, sensor_type, device), group in groups:
        group = group.sort_values('timestamp').reset_index(drop=True)
        activity_label = ACTIVITY_MAPPING[activity_code]
        
        n_samples = len(group)
        start_idx = 0
        
        while start_idx < n_samples:
            end_idx = min(start_idx + WINDOW_SIZE, n_samples)
            window_data = group.iloc[start_idx:end_idx]
            actual_size = len(window_data)
            
            # Handle short segments with mean padding
            if actual_size < WINDOW_SIZE:
                if actual_size < WINDOW_SIZE // 2:
                    break
                
                x_vals = list(window_data['x'].values)
                y_vals = list(window_data['y'].values)
                z_vals = list(window_data['z'].values)
                
                padding_needed = WINDOW_SIZE - actual_size
                x_vals.extend([np.mean(x_vals)] * padding_needed)
                y_vals.extend([np.mean(y_vals)] * padding_needed)
                z_vals.extend([np.mean(z_vals)] * padding_needed)
                
                windows_padded += 1
            else:
                x_vals = window_data['x'].values[:WINDOW_SIZE]
                y_vals = window_data['y'].values[:WINDOW_SIZE]
                z_vals = window_data['z'].values[:WINDOW_SIZE]
            
            # Create window row
            window_row = {}
            for i in range(WINDOW_SIZE):
                window_row[f'x_{i}'] = x_vals[i]
                window_row[f'y_{i}'] = y_vals[i]
                window_row[f'z_{i}'] = z_vals[i]
            
            window_row['subject_id'] = subject_id
            window_row['activity_label'] = activity_label
            window_row['sensor_type'] = sensor_type
            window_row['device'] = device
            
            windows.append(window_row)
            start_idx += HOP_SIZE
    
    windows_df = pd.DataFrame(windows)
    
    logger.log(f"Total windows created: {len(windows_df):,}")
    logger.log(f"Windows padded: {windows_padded}")
    
    # Log windows per activity
    activity_counts = windows_df['activity_label'].value_counts().sort_index()
    for label, count in activity_counts.items():
        logger.log(f"  Activity {label} ({ACTIVITY_NAMES[label]}): {count:,} windows")
    
    return windows_df


def run(df=None):
    """Execute Step 2: Create windows."""
    logger.header("STEP 2: Windowing")
    
    if df is None:
        df = pd.read_csv(os.path.join(DATA_DIR, '01_cleaned', 'cleaned_data.csv'))
    
    windows_df = create_windows(df)
    
    # Save windowed data
    output_dir = os.path.join(DATA_DIR, '02_windowed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'windowed_data.csv')
    windows_df.to_csv(output_path, index=False)
    
    logger.log(f"Saved windowed data to {output_path}")
    logger.log(f"Final shape: {windows_df.shape}")
    logger.log("")
    
    return windows_df


if __name__ == '__main__':
    run()
