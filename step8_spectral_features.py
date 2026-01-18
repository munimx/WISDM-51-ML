"""
WISDM-51 Activity Recognition Pipeline
Step 8: Spectral (Frequency-Domain) Feature Extraction

Computes spectral features from windowed sensor data using FFT.
Extracts 13 spectral features per axis (x, y, z) = 39 features total per window.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

from config import (DATA_DIR, VIS_DIR, WINDOW_SIZE, SAMPLING_RATE, 
                    METADATA_COLS, SPECTRAL_ROLLOFF_THRESHOLD, FREQ_BANDS)
from logger import logger


def compute_spectral_features(signal, sampling_rate=SAMPLING_RATE):
    """
    Compute all spectral features for a single axis.
    
    Parameters:
    -----------
    signal : array-like
        Time-domain signal (60 samples)
    sampling_rate : int
        Sampling rate in Hz (default: 20)
    
    Returns:
    --------
    dict : Dictionary of spectral features (13 features)
    """
    features = {}
    signal = np.array(signal)
    N = len(signal)
    
    # Compute FFT
    fft_vals = fft(signal)
    fft_freqs = fftfreq(N, 1/sampling_rate)
    
    # Use only positive frequencies
    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    magnitudes = np.abs(fft_vals[pos_mask])
    
    # Normalize magnitudes for probability distribution
    mag_sum = np.sum(magnitudes)
    if mag_sum > 0:
        psd = magnitudes / mag_sum
    else:
        psd = np.zeros_like(magnitudes)
    
    # 1. Spectral Energy - Sum of squared magnitudes
    features['spectral_energy'] = np.sum(magnitudes ** 2)
    
    # 2. Spectral Entropy - Measure of spectral complexity
    psd_nonzero = psd[psd > 0]
    if len(psd_nonzero) > 0:
        features['spectral_entropy'] = -np.sum(psd_nonzero * np.log2(psd_nonzero))
    else:
        features['spectral_entropy'] = 0
    
    # 3. Spectral Centroid - Center of mass of spectrum
    if mag_sum > 0:
        features['spectral_centroid'] = np.sum(freqs * psd)
    else:
        features['spectral_centroid'] = 0
    
    # 4. Spectral Spread - Std dev around centroid
    centroid = features['spectral_centroid']
    if mag_sum > 0:
        features['spectral_spread'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd))
    else:
        features['spectral_spread'] = 0
    
    # 5. Spectral Flux - Rate of change
    features['spectral_flux'] = np.sqrt(np.sum(magnitudes ** 2))
    
    # 6. Spectral Roll-off - Freq below which 85% energy lies
    if len(magnitudes) > 0 and np.sum(magnitudes) > 0:
        cumsum = np.cumsum(magnitudes)
        rolloff_threshold = SPECTRAL_ROLLOFF_THRESHOLD * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = freqs[rolloff_idx[0]]
        else:
            features['spectral_rolloff'] = freqs[-1] if len(freqs) > 0 else 0
    else:
        features['spectral_rolloff'] = 0
    
    # 7. Spectral Flatness - Tonality vs noise
    if len(magnitudes) > 0:
        geometric_mean = np.exp(np.mean(np.log(magnitudes + 1e-10)))
        arithmetic_mean = np.mean(magnitudes)
        features['spectral_flatness'] = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
    else:
        features['spectral_flatness'] = 0
    
    # 8. Dominant Frequency - Freq with highest magnitude
    if len(magnitudes) > 0:
        dominant_idx = np.argmax(magnitudes)
        features['dominant_frequency'] = freqs[dominant_idx]
        features['dominant_amplitude'] = magnitudes[dominant_idx]
    else:
        features['dominant_frequency'] = 0
        features['dominant_amplitude'] = 0
    
    # 9. Bandpower features
    band1_mask = (freqs >= FREQ_BANDS['low'][0]) & (freqs < FREQ_BANDS['low'][1])
    band2_mask = (freqs >= FREQ_BANDS['mid'][0]) & (freqs < FREQ_BANDS['mid'][1])
    features['bandpower_0_5hz'] = np.sum(magnitudes[band1_mask] ** 2)
    features['bandpower_5_10hz'] = np.sum(magnitudes[band2_mask] ** 2)
    
    # 10. Periodicity - Autocorrelation-based
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    if len(autocorr) > 1 and autocorr[0] != 0:
        features['periodicity'] = autocorr[1] / autocorr[0]
    else:
        features['periodicity'] = 0
    
    # 11. Harmonic Ratio - Using peaks
    peaks, _ = find_peaks(magnitudes, height=np.mean(magnitudes) * 0.5)
    if len(peaks) > 0:
        harmonic_power = np.sum(magnitudes[peaks] ** 2)
        total_power = np.sum(magnitudes ** 2)
        features['harmonic_ratio'] = harmonic_power / total_power if total_power > 0 else 0
    else:
        features['harmonic_ratio'] = 0
    
    return features


def extract_all_spectral_features(row):
    """Extract spectral features for all 3 axes from a single window row."""
    features = {}
    
    for axis in ['x', 'y', 'z']:
        # Get axis data from window
        axis_data = [row[f'{axis}_{i}'] for i in range(WINDOW_SIZE)]
        
        # Compute spectral features
        axis_features = compute_spectral_features(axis_data)
        
        # Add axis suffix to feature names
        for feat_name, feat_val in axis_features.items():
            features[f'{feat_name}_{axis}'] = feat_val
    
    return features


def generate_spectral_visualizations(df, feature_cols):
    """Generate visualizations for spectral features."""
    vis_dir = os.path.join(VIS_DIR, 'spectral_features')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Select subset of features for visualization (first 15)
    selected_features = feature_cols[:15]
    
    # Histogram plot
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(selected_features):
        if i < len(axes):
            data = df[col].dropna()
            axes[i].hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(labelsize=7)
    
    plt.suptitle('Spectral Feature Distributions (Sample)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'spectral_features_histogram.png'), dpi=300)
    plt.close()
    
    # Boxplot
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(selected_features):
        if i < len(axes):
            data = df[col].dropna()
            axes[i].boxplot(data)
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(labelsize=7)
    
    plt.suptitle('Spectral Feature Distributions (Boxplot)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'spectral_features_boxplot.png'), dpi=300)
    plt.close()
    
    logger.log("Saved spectral feature distribution visualizations")


def generate_sample_spectrum_plot(df):
    """Generate sample spectrum plots for different activity types."""
    vis_dir = os.path.join(VIS_DIR, 'spectral_features')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Sample activities: static (sitting=3), dynamic (walking=0), transition (stairs=2)
    sample_activities = {
        'Static (Sitting)': 3,
        'Dynamic (Walking)': 0,
        'Transition (Stairs)': 2
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (activity_name, activity_label) in enumerate(sample_activities.items()):
        # Get a sample window for this activity
        activity_df = df[df['activity_label'] == activity_label]
        if len(activity_df) > 0:
            sample_row = activity_df.iloc[0]
            
            # Get x-axis data
            signal = [sample_row[f'x_{i}'] for i in range(WINDOW_SIZE)]
            signal = np.array(signal)
            
            # Compute FFT
            fft_vals = fft(signal)
            fft_freqs = fftfreq(len(signal), 1/SAMPLING_RATE)
            
            # Positive frequencies only
            pos_mask = fft_freqs > 0
            freqs = fft_freqs[pos_mask]
            magnitudes = np.abs(fft_vals[pos_mask])
            
            axes[idx].plot(freqs, magnitudes, color='steelblue', linewidth=1.5)
            axes[idx].fill_between(freqs, magnitudes, alpha=0.3)
            axes[idx].set_xlabel('Frequency (Hz)')
            axes[idx].set_ylabel('Magnitude')
            axes[idx].set_title(activity_name)
            axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Sample Frequency Spectra by Activity Type (X-axis)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'sample_spectrum_plot.png'), dpi=300)
    plt.close()
    
    logger.log("Saved sample spectrum plots")


def run(windowed_df=None):
    """Execute Step 8: Extract spectral features from windowed data."""
    logger.header("STEP 8: Spectral Feature Extraction")
    
    # Load windowed data if not provided
    if windowed_df is None:
        windowed_path = os.path.join(DATA_DIR, '02_windowed', 'windowed_data.csv')
        logger.log(f"Loading windowed data from {windowed_path}")
        windowed_df = pd.read_csv(windowed_path)
    
    logger.log(f"Processing {len(windowed_df):,} windows...")
    
    # Generate sample spectrum plot before feature extraction
    generate_sample_spectrum_plot(windowed_df)
    
    # Extract spectral features for each window
    features_list = []
    total_windows = len(windowed_df)
    
    for idx, row in windowed_df.iterrows():
        # Extract spectral features
        features = extract_all_spectral_features(row)
        
        # Add metadata
        features['subject_id'] = row['subject_id']
        features['activity_label'] = row['activity_label']
        features['sensor_type'] = row['sensor_type']
        features['device'] = row['device']
        
        features_list.append(features)
        
        if (idx + 1) % 50000 == 0:
            logger.log(f"  Processed {idx + 1:,}/{total_windows:,} windows...")
    
    # Create DataFrame
    spectral_df = pd.DataFrame(features_list)
    
    # Get feature columns (excluding metadata)
    feature_cols = [c for c in spectral_df.columns if c not in METADATA_COLS]
    
    logger.log(f"Extracted {len(feature_cols)} spectral features")
    
    # Handle NaN/Inf values
    spectral_df[feature_cols] = spectral_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    spectral_df[feature_cols] = spectral_df[feature_cols].fillna(0)
    
    # Save spectral features
    output_dir = os.path.join(DATA_DIR, '08_spectral')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'SPECTRAL_FEATURES.csv')
    spectral_df.to_csv(output_path, index=False)
    
    logger.log(f"Saved spectral features to {output_path}")
    logger.log(f"Final shape: {spectral_df.shape}")
    
    # Generate visualizations
    logger.log("Generating spectral feature visualizations...")
    generate_spectral_visualizations(spectral_df, feature_cols)
    
    # Save feature descriptions
    desc_path = os.path.join(output_dir, 'spectral_feature_descriptions.txt')
    with open(desc_path, 'w') as f:
        f.write("WISDM-51 Spectral Feature Descriptions\n")
        f.write("=" * 50 + "\n\n")
        f.write("Features extracted per axis (x, y, z):\n\n")
        f.write("1. spectral_energy - Sum of squared magnitudes in frequency domain\n")
        f.write("2. spectral_entropy - Measure of spectral complexity/randomness\n")
        f.write("3. spectral_centroid - Center of mass of the spectrum\n")
        f.write("4. spectral_spread - Standard deviation around spectral centroid\n")
        f.write("5. spectral_flux - Rate of change in spectrum\n")
        f.write("6. spectral_rolloff - Frequency below which 85% of energy lies\n")
        f.write("7. spectral_flatness - Ratio of geometric to arithmetic mean\n")
        f.write("8. dominant_frequency - Frequency with highest magnitude\n")
        f.write("9. dominant_amplitude - Magnitude at dominant frequency\n")
        f.write("10. bandpower_0_5hz - Power in 0-5 Hz band\n")
        f.write("11. bandpower_5_10hz - Power in 5-10 Hz band\n")
        f.write("12. periodicity - Autocorrelation-based measure\n")
        f.write("13. harmonic_ratio - Ratio of harmonic to total power\n")
        f.write(f"\nTotal features: {len(feature_cols)}\n")
    
    logger.log(f"Saved feature descriptions to {desc_path}")
    logger.log("")
    
    return spectral_df


if __name__ == '__main__':
    run()
