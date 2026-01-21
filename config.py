"""
WISDM-51 Activity Recognition Pipeline - Configuration
Shared constants and settings used across all pipeline steps.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'raw')
DATA_DIR = os.path.join(BASE_DIR, 'data')
VIS_DIR = os.path.join(BASE_DIR, 'visualizations')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Activity mapping
ACTIVITY_MAPPING = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
    'M': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17
}

ACTIVITY_NAMES = {
    'A': 'Walking', 'B': 'Jogging', 'C': 'Stairs', 'D': 'Sitting', 
    'E': 'Standing', 'F': 'Typing', 'G': 'Brushing Teeth', 'H': 'Eating Soup',
    'I': 'Eating Chips', 'J': 'Eating Pasta', 'K': 'Drinking', 
    'L': 'Eating Sandwich', 'M': 'Kicking', 'O': 'Playing Catch',
    'P': 'Dribbling', 'Q': 'Writing', 'R': 'Clapping', 'S': 'Folding Clothes',
    # Numeric labels
    0: 'Walking', 1: 'Jogging', 2: 'Stairs', 3: 'Sitting', 
    4: 'Standing', 5: 'Typing', 6: 'Brushing Teeth', 7: 'Eating Soup',
    8: 'Eating Chips', 9: 'Eating Pasta', 10: 'Drinking', 
    11: 'Eating Sandwich', 12: 'Kicking', 13: 'Playing Catch',
    14: 'Dribbling', 15: 'Writing', 16: 'Clapping', 17: 'Folding Clothes'
}

# Signal processing parameters
SAMPLING_RATE = 20      # Hz
WINDOW_SIZE = 60        # 3 seconds at 20 Hz
HOP_SIZE = 30           # 50% overlap

# Scalers to use (reduced set)
SCALER_NAMES = ['minmax', 'standard', 'robust']

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
NUM_FEATURES = 50  # Increased for combined features

# Spectral feature parameters
SPECTRAL_ROLLOFF_THRESHOLD = 0.85  # 85% energy threshold
SPECTRAL_PEAK_PROMINENCE = 0.5     # Relative to mean magnitude
NUM_SPECTRAL_FEATURES = 30         # Features to select for spectral

# Frequency bands (Hz) - Nyquist frequency = sampling_rate/2 = 10Hz
FREQ_BANDS = {
    'low': (0, 5),
    'mid': (5, 10),
}

# Advanced feature parameters
USE_ADVANCED_FEATURES = True
USE_WAVELET_FEATURES = True
USE_ENTROPY_FEATURES = True

# Optimization parameters
USE_PARALLEL_PROCESSING = True
N_JOBS = -1  # Use all CPU cores
CHUNK_SIZE = 50000  # For batch processing

# Model hyperparameters
CV_FOLDS = 5
GRID_SEARCH_VERBOSE = 1

# Ensemble parameters
VOTING_WEIGHTS = [1, 1, 1]  # Equal weights for voting
STACKING_CV = 5

# Metadata columns (activity_label is numeric 0-17)
METADATA_COLS = ['subject_id', 'activity_label', 'sensor_type', 'device']
