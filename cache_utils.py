"""
WISDM-51 Activity Recognition Pipeline - Caching Utilities
Provides checkpoint/resume functionality for long-running operations.
"""

import os
import hashlib
import pickle
import pandas as pd
from config import CACHE_DIR, ENABLE_CACHING


def get_cache_key(*args):
    """Generate cache key from arguments."""
    key_str = str(args)
    return hashlib.md5(key_str.encode()).hexdigest()


def cache_result(key, data, cache_type='pickle'):
    """Cache result to disk."""
    if not ENABLE_CACHING:
        return
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{key}.{cache_type}")
    
    if cache_type == 'pickle':
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    elif cache_type == 'csv':
        data.to_csv(cache_path, index=False)
    elif cache_type == 'pkl':
        import joblib
        joblib.dump(data, cache_path)


def load_cached_result(key, cache_type='pickle'):
    """Load cached result from disk."""
    if not ENABLE_CACHING:
        return None
    
    cache_path = os.path.join(CACHE_DIR, f"{key}.{cache_type}")
    
    if not os.path.exists(cache_path):
        return None
    
    if cache_type == 'pickle':
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    elif cache_type == 'csv':
        return pd.read_csv(cache_path)
    elif cache_type == 'pkl':
        import joblib
        return joblib.load(cache_path)


def clear_cache():
    """Clear all cached results."""
    if os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cache cleared: {CACHE_DIR}")
