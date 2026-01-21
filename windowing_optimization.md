# WISDM-51 Windowing Performance Optimization Guide

## Problem Analysis

The current `step2_windowing.py` implementation is slow due to:

1. **Inefficient iteration**: Using `iterrows()` and `groupby()` with manual window creation
2. **List concatenation in loops**: Extending lists repeatedly in tight loops
3. **Dictionary creation per window**: Creating dictionaries for each window individually
4. **No vectorization**: Not leveraging pandas/numpy vectorized operations

## Optimization Strategy

Replace the current manual windowing approach with **vectorized sliding window generation** using numpy's stride tricks or pandas rolling operations.

---

## OPTIMIZED IMPLEMENTATION

Replace the entire `create_windows()` function in `step2_windowing.py` with the following optimized version:

### Option 1: NumPy Stride-Based (FASTEST - Recommended)

```python
def create_windows_vectorized(df):
    """Create sliding windows using vectorized operations - OPTIMIZED VERSION."""
    from numpy.lib.stride_tricks import sliding_window_view
    
    windows_list = []
    windows_padded = 0
    
    # Group by subject, activity, sensor, device
    groups = df.groupby(['subject_id', 'activity_code', 'sensor_type', 'device'])
    total_groups = len(groups)
    logger.log(f"Processing {total_groups} groups...")
    
    group_counter = 0
    for (subject_id, activity_code, sensor_type, device), group in groups:
        group = group.sort_values('timestamp').reset_index(drop=True)
        activity_label = ACTIVITY_MAPPING[activity_code]
        n_samples = len(group)
        
        # Skip if too short
        if n_samples < WINDOW_SIZE // 2:
            continue
        
        # Extract xyz arrays
        x_vals = group['x'].values
        y_vals = group['y'].values
        z_vals = group['z'].values
        
        # Handle padding if needed
        if n_samples < WINDOW_SIZE:
            padding_needed = WINDOW_SIZE - n_samples
            x_vals = np.concatenate([x_vals, [np.mean(x_vals)] * padding_needed])
            y_vals = np.concatenate([y_vals, [np.mean(y_vals)] * padding_needed])
            z_vals = np.concatenate([z_vals, [np.mean(z_vals)] * padding_needed])
            windows_padded += 1
            n_samples = WINDOW_SIZE
        
        # Create sliding windows using stride tricks (FAST!)
        if n_samples >= WINDOW_SIZE:
            # Calculate number of windows
            n_windows = (n_samples - WINDOW_SIZE) // HOP_SIZE + 1
            
            # Create windows for each axis
            x_windows = sliding_window_view(x_vals, WINDOW_SIZE)[::HOP_SIZE][:n_windows]
            y_windows = sliding_window_view(y_vals, WINDOW_SIZE)[::HOP_SIZE][:n_windows]
            z_windows = sliding_window_view(z_vals, WINDOW_SIZE)[::HOP_SIZE][:n_windows]
            
            # Create DataFrame columns efficiently
            window_data = {}
            
            # Add x, y, z columns
            for i in range(WINDOW_SIZE):
                window_data[f'x_{i}'] = x_windows[:, i]
                window_data[f'y_{i}'] = y_windows[:, i]
                window_data[f'z_{i}'] = z_windows[:, i]
            
            # Add metadata (broadcast to all windows)
            window_data['subject_id'] = subject_id
            window_data['activity_label'] = activity_label
            window_data['sensor_type'] = sensor_type
            window_data['device'] = device
            
            # Create DataFrame for this group's windows
            group_df = pd.DataFrame(window_data)
            windows_list.append(group_df)
        
        group_counter += 1
        if group_counter % 100 == 0:
            logger.log(f"  Processed {group_counter}/{total_groups} groups...")
    
    # Concatenate all windows
    if windows_list:
        windows_df = pd.concat(windows_list, ignore_index=True)
    else:
        windows_df = pd.DataFrame()
    
    logger.log(f"Total windows created: {len(windows_df):,}")
    logger.log(f"Windows padded: {windows_padded}")
    
    # Log windows per activity
    if len(windows_df) > 0:
        activity_counts = windows_df['activity_label'].value_counts().sort_index()
        for label, count in activity_counts.items():
            logger.log(f"  Activity {label} ({ACTIVITY_NAMES[label]}): {count:,} windows")
    
    return windows_df
```

### Option 2: Pandas Rolling (SIMPLER, Still Fast)

```python
def create_windows_pandas_rolling(df):
    """Create sliding windows using pandas rolling - SIMPLER ALTERNATIVE."""
    windows_list = []
    windows_padded = 0
    
    groups = df.groupby(['subject_id', 'activity_code', 'sensor_type', 'device'])
    total_groups = len(groups)
    logger.log(f"Processing {total_groups} groups...")
    
    group_counter = 0
    for (subject_id, activity_code, sensor_type, device), group in groups:
        group = group.sort_values('timestamp').reset_index(drop=True)
        activity_label = ACTIVITY_MAPPING[activity_code]
        n_samples = len(group)
        
        if n_samples < WINDOW_SIZE // 2:
            continue
        
        # Handle padding
        if n_samples < WINDOW_SIZE:
            padding_needed = WINDOW_SIZE - n_samples
            pad_data = {
                'x': [group['x'].mean()] * padding_needed,
                'y': [group['y'].mean()] * padding_needed,
                'z': [group['z'].mean()] * padding_needed,
                'timestamp': [group['timestamp'].iloc[-1]] * padding_needed,
                'subject_id': [subject_id] * padding_needed,
                'activity_code': [activity_code] * padding_needed,
                'sensor_type': [sensor_type] * padding_needed,
                'device': [device] * padding_needed
            }
            group = pd.concat([group, pd.DataFrame(pad_data)], ignore_index=True)
            windows_padded += 1
            n_samples = len(group)
        
        # Create windows using iloc with step
        window_starts = range(0, n_samples - WINDOW_SIZE + 1, HOP_SIZE)
        
        for start_idx in window_starts:
            end_idx = start_idx + WINDOW_SIZE
            window_data = group.iloc[start_idx:end_idx]
            
            # Create window row efficiently
            window_row = {
                'subject_id': subject_id,
                'activity_label': activity_label,
                'sensor_type': sensor_type,
                'device': device
            }
            
            # Add xyz values
            for i, (x, y, z) in enumerate(zip(window_data['x'], window_data['y'], window_data['z'])):
                window_row[f'x_{i}'] = x
                window_row[f'y_{i}'] = y
                window_row[f'z_{i}'] = z
            
            windows_list.append(window_row)
        
        group_counter += 1
        if group_counter % 100 == 0:
            logger.log(f"  Processed {group_counter}/{total_groups} groups...")
    
    windows_df = pd.DataFrame(windows_list)
    
    logger.log(f"Total windows created: {len(windows_df):,}")
    logger.log(f"Windows padded: {windows_padded}")
    
    if len(windows_df) > 0:
        activity_counts = windows_df['activity_label'].value_counts().sort_index()
        for label, count in activity_counts.items():
            logger.log(f"  Activity {label} ({ACTIVITY_NAMES[label]}): {count:,} windows")
    
    return windows_df
```

### Option 3: Ultra-Fast NumPy Only (MAXIMUM SPEED)

```python
def create_windows_numpy_ultra_fast(df):
    """Ultra-fast windowing using pure NumPy - MAXIMUM PERFORMANCE."""
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Pre-allocate result arrays
    groups = df.groupby(['subject_id', 'activity_code', 'sensor_type', 'device'])
    
    # Estimate total windows for pre-allocation
    estimated_windows = 0
    for _, group in groups:
        n = len(group)
        if n >= WINDOW_SIZE // 2:
            estimated_windows += max(1, (n - WINDOW_SIZE) // HOP_SIZE + 1)
    
    logger.log(f"Estimated windows: {estimated_windows:,}")
    
    # Pre-allocate arrays
    xyz_data = np.zeros((estimated_windows, WINDOW_SIZE * 3), dtype=np.float32)
    metadata = np.zeros((estimated_windows, 4), dtype=np.int32)  # subject, activity, sensor_id, device_id
    
    window_idx = 0
    windows_padded = 0
    
    # Create encoding maps
    sensor_map = {'accel': 0, 'gyro': 1}
    device_map = {'phone': 0, 'watch': 1}
    
    for (subject_id, activity_code, sensor_type, device), group in groups:
        group = group.sort_values('timestamp')
        n_samples = len(group)
        
        if n_samples < WINDOW_SIZE // 2:
            continue
        
        x = group['x'].values.astype(np.float32)
        y = group['y'].values.astype(np.float32)
        z = group['z'].values.astype(np.float32)
        
        # Padding if needed
        if n_samples < WINDOW_SIZE:
            pad_len = WINDOW_SIZE - n_samples
            x = np.concatenate([x, np.full(pad_len, x.mean())])
            y = np.concatenate([y, np.full(pad_len, y.mean())])
            z = np.concatenate([z, np.full(pad_len, z.mean())])
            windows_padded += 1
            n_samples = WINDOW_SIZE
        
        # Create windows
        if n_samples >= WINDOW_SIZE:
            n_windows = (n_samples - WINDOW_SIZE) // HOP_SIZE + 1
            
            x_win = sliding_window_view(x, WINDOW_SIZE)[::HOP_SIZE][:n_windows]
            y_win = sliding_window_view(y, WINDOW_SIZE)[::HOP_SIZE][:n_windows]
            z_win = sliding_window_view(z, WINDOW_SIZE)[::HOP_SIZE][:n_windows]
            
            # Stack xyz horizontally
            batch_size = len(x_win)
            xyz_data[window_idx:window_idx+batch_size] = np.hstack([x_win, y_win, z_win])
            
            # Metadata
            metadata[window_idx:window_idx+batch_size] = [
                subject_id,
                ACTIVITY_MAPPING[activity_code],
                sensor_map[sensor_type],
                device_map[device]
            ]
            
            window_idx += batch_size
    
    # Trim to actual size
    xyz_data = xyz_data[:window_idx]
    metadata = metadata[:window_idx]
    
    logger.log(f"Total windows created: {window_idx:,}")
    logger.log(f"Windows padded: {windows_padded}")
    
    # Convert to DataFrame
    columns = {}
    for i in range(WINDOW_SIZE):
        columns[f'x_{i}'] = xyz_data[:, i]
        columns[f'y_{i}'] = xyz_data[:, WINDOW_SIZE + i]
        columns[f'z_{i}'] = xyz_data[:, 2*WINDOW_SIZE + i]
    
    # Decode metadata
    reverse_sensor = {0: 'accel', 1: 'gyro'}
    reverse_device = {0: 'phone', 1: 'watch'}
    
    columns['subject_id'] = metadata[:, 0]
    columns['activity_label'] = metadata[:, 1]
    columns['sensor_type'] = [reverse_sensor[x] for x in metadata[:, 2]]
    columns['device'] = [reverse_device[x] for x in metadata[:, 3]]
    
    windows_df = pd.DataFrame(columns)
    
    if len(windows_df) > 0:
        activity_counts = windows_df['activity_label'].value_counts().sort_index()
        for label, count in activity_counts.items():
            logger.log(f"  Activity {label} ({ACTIVITY_NAMES[label]}): {count:,} windows")
    
    return windows_df
```

---

## IMPLEMENTATION INSTRUCTIONS

### Step 1: Update step2_windowing.py

Replace the `create_windows()` function with **Option 1** (recommended for best balance of speed and readability).

### Step 2: Update the run() function

```python
def run(df=None):
    """Execute Step 2: Create windows."""
    logger.header("STEP 2: Windowing")
    
    if df is None:
        df = pd.read_csv(os.path.join(DATA_DIR, '01_cleaned', 'cleaned_data.csv'))
    
    # Use optimized windowing function
    windows_df = create_windows_vectorized(df)  # <-- Change this line
    
    # Save windowed data
    output_dir = os.path.join(DATA_DIR, '02_windowed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'windowed_data.csv')
    windows_df.to_csv(output_path, index=False)
    
    logger.log(f"Saved windowed data to {output_path}")
    logger.log(f"Final shape: {windows_df.shape}")
    logger.log("")
    
    return windows_df
```

### Step 3: Add imports at top of file

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
```

---

## PERFORMANCE COMPARISON

| Method | Estimated Time (100K samples) | Memory Usage |
|--------|-------------------------------|--------------|
| **Original (iterrows)** | ~5-10 minutes | Low |
| **Option 1: NumPy Stride** | ~10-30 seconds | Medium |
| **Option 2: Pandas Rolling** | ~1-2 minutes | Medium |
| **Option 3: Ultra-Fast NumPy** | ~5-15 seconds | High |

**Expected Speedup: 10-40x faster**

---

## ADDITIONAL OPTIMIZATIONS

### 1. Parallel Processing for Large Datasets

```python
from multiprocessing import Pool, cpu_count

def process_group_parallel(args):
    """Process a single group in parallel."""
    (subject_id, activity_code, sensor_type, device), group_data = args
    # ... windowing logic here
    return windows_df

def create_windows_parallel(df):
    """Create windows using parallel processing."""
    groups = list(df.groupby(['subject_id', 'activity_code', 'sensor_type', 'device']))
    
    with Pool(cpu_count() - 1) as pool:
        results = pool.map(process_group_parallel, groups)
    
    return pd.concat(results, ignore_index=True)
```

### 2. Chunked Processing for Memory Efficiency

```python
def create_windows_chunked(df, chunk_size=10000):
    """Process in chunks to reduce memory usage."""
    groups = df.groupby(['subject_id', 'activity_code', 'sensor_type', 'device'])
    
    output_path = os.path.join(DATA_DIR, '02_windowed', 'windowed_data.csv')
    first_chunk = True
    
    for i, ((subject_id, activity_code, sensor_type, device), group) in enumerate(groups):
        # Process group...
        group_windows = process_single_group(group, subject_id, activity_code, sensor_type, device)
        
        # Write to CSV incrementally
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        group_windows.to_csv(output_path, mode=mode, header=header, index=False)
        first_chunk = False
```

### 3. Use Numba JIT Compilation (Advanced)

```python
from numba import jit

@jit(nopython=True)
def create_windows_numba(x, y, z, window_size, hop_size):
    """JIT-compiled windowing for maximum speed."""
    n = len(x)
    n_windows = (n - window_size) // hop_size + 1
    
    x_windows = np.zeros((n_windows, window_size))
    y_windows = np.zeros((n_windows, window_size))
    z_windows = np.zeros((n_windows, window_size))
    
    for i in range(n_windows):
        start = i * hop_size
        x_windows[i] = x[start:start+window_size]
        y_windows[i] = y[start:start+window_size]
        z_windows[i] = z[start:start+window_size]
    
    return x_windows, y_windows, z_windows
```

---

## TROUBLESHOOTING

### Issue: "ImportError: cannot import name 'sliding_window_view'"

**Solution:** Update NumPy to >= 1.20.0
```bash
pip install numpy>=1.20.0
```

### Issue: Memory Error with Large Datasets

**Solution:** Use chunked processing (see optimization #2 above)

### Issue: Results differ slightly from original

**Cause:** Floating-point precision differences
**Solution:** This is normal and acceptable; differences should be < 1e-6

---

## VERIFICATION

After implementing, verify correctness:

```python
# Test with small sample
test_df = df.head(1000)
original_windows = create_windows(test_df)  # Old method
optimized_windows = create_windows_vectorized(test_df)  # New method

# Compare shapes
assert original_windows.shape == optimized_windows.shape

# Compare values (allow small floating-point errors)
for col in original_windows.columns:
    if col not in ['subject_id', 'activity_label', 'sensor_type', 'device']:
        np.testing.assert_allclose(
            original_windows[col].values,
            optimized_windows[col].values,
            rtol=1e-5
        )
```

---

## RECOMMENDATION

**Use Option 1 (NumPy Stride-Based)** for the best balance of:
- ✅ Performance (10-30x speedup)
- ✅ Readability
- ✅ Memory efficiency
- ✅ Maintainability

Expected time reduction: **5-10 minutes → 10-30 seconds**

---

## SUMMARY

The optimization leverages:
1. **Vectorized operations** instead of row-by-row iteration
2. **NumPy stride tricks** for zero-copy window creation
3. **Batch DataFrame construction** instead of list appends
4. **Efficient memory layout** with contiguous arrays

This should reduce windowing time by **10-40x** depending on dataset size.
