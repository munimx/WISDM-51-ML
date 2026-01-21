"""
WISDM-51 Activity Recognition Pipeline
Main Runner Script

Executes all pipeline steps in sequence or individual steps as specified.
"""

import sys
import time
from datetime import datetime

from logger import logger
import step1_data_cleaning
import step2_windowing
import step3_feature_extraction
import step3b_advanced_features
import step3c_combine_features
import step4_scaling
import step5_feature_selection
import step6_model_training
import step6b_optimized_models
import step6c_ensemble_models
import step7_results_summary
import step8_spectral_features
import step9_spectral_scaling
import step10_spectral_selection
import step11_spectral_model_training
import step12_final_comparison


def run_full_pipeline():
    """Execute the complete pipeline from start to finish."""
    start_time = time.time()
    
    logger.header("WISDM-51 ACTIVITY RECOGNITION PIPELINE")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("")
    
    # Part 1: Time-Domain Features
    logger.header("PART 1: TIME-DOMAIN FEATURES")
    
    # Step 1: Data Cleaning
    cleaned_df = step1_data_cleaning.run()
    
    # Step 2: Windowing
    windowed_df = step2_windowing.run(cleaned_df)
    
    # Step 3: Feature Extraction
    features_df = step3_feature_extraction.run(windowed_df)
    
    # Step 4: Scaling
    scaled_dfs = step4_scaling.run(features_df)
    
    # Step 5: Feature Selection
    selected_dfs = step5_feature_selection.run(scaled_dfs)
    
    # Step 6: Model Training
    results_df = step6_model_training.run(selected_dfs)
    
    # Step 7: Results Summary
    step7_results_summary.run(results_df)
    
    # Part 2: Spectral Features
    logger.header("PART 2: SPECTRAL (FREQUENCY-DOMAIN) FEATURES")
    
    # Step 8: Spectral Feature Extraction
    spectral_df = step8_spectral_features.run(windowed_df)
    
    # Step 9: Spectral Scaling
    scaled_spectral_df = step9_spectral_scaling.run(spectral_df)
    
    # Step 10: Spectral Feature Selection
    selected_spectral_df = step10_spectral_selection.run(scaled_spectral_df)
    
    # Step 11: Spectral Model Training
    spectral_results_df = step11_spectral_model_training.run(selected_spectral_df)
    
    # Step 12: Final Comparison
    step12_final_comparison.run(spectral_results_df)
    
    elapsed = time.time() - start_time
    logger.header("PIPELINE COMPLETE")
    logger.log(f"Total time: {elapsed/60:.2f} minutes")
    logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def run_optimized_pipeline():
    """Execute the optimized pipeline with combined features and ensemble models."""
    start_time = time.time()
    
    logger.header("WISDM-51 OPTIMIZED PIPELINE")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("Target: 80%+ accuracy with combined features and ensembles")
    logger.log("")
    
    # Part 1: Data Preparation (reuse existing windowed data if available)
    import os
    from config import DATA_DIR
    windowed_path = os.path.join(DATA_DIR, '02_windowed', 'windowed_data.csv')
    
    if os.path.exists(windowed_path):
        logger.log("Loading existing windowed data...")
        import pandas as pd
        windowed_df = pd.read_csv(windowed_path)
    else:
        logger.header("PREPARING DATA")
        cleaned_df = step1_data_cleaning.run()
        windowed_df = step2_windowing.run(cleaned_df)
    
    # Part 2: Basic Feature Extraction (reuse if available)
    basic_features_path = os.path.join(DATA_DIR, '03_features', 'features.csv')
    if os.path.exists(basic_features_path):
        logger.log("Loading existing basic features...")
        import pandas as pd
        basic_features_df = pd.read_csv(basic_features_path)
    else:
        basic_features_df = step3_feature_extraction.run(windowed_df)
    
    # Part 3: Spectral Features (reuse if available)
    spectral_path = os.path.join(DATA_DIR, '08_spectral_features', 'spectral_features.csv')
    if os.path.exists(spectral_path):
        logger.log("Loading existing spectral features...")
        import pandas as pd
        spectral_df = pd.read_csv(spectral_path)
    else:
        spectral_df = step8_spectral_features.run(windowed_df)
    
    # Part 4: Advanced Features
    logger.header("PART 3: ADVANCED FEATURE EXTRACTION")
    advanced_df = step3b_advanced_features.run(windowed_df)
    
    # Part 5: Combine All Features
    combined_df = step3c_combine_features.run(basic_features_df, advanced_df, spectral_df)
    
    # Part 6: Scaling (on combined features)
    logger.header("PART 4: SCALING COMBINED FEATURES")
    scaled_dfs = step4_scaling.run(combined_df)
    
    # Part 7: Feature Selection
    logger.header("PART 5: FEATURE SELECTION")
    selected_dfs = step5_feature_selection.run(scaled_dfs)
    
    # Part 8: Optimized Model Training
    logger.header("PART 6: OPTIMIZED MODEL TRAINING")
    optimized_results = step6b_optimized_models.run()
    
    # Part 9: Ensemble Models
    logger.header("PART 7: ENSEMBLE MODELS")
    ensemble_results = step6c_ensemble_models.run()
    
    elapsed = time.time() - start_time
    logger.header("OPTIMIZED PIPELINE COMPLETE")
    logger.log(f"Total time: {elapsed/60:.2f} minutes")
    logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def run_step(step_num):
    """Execute a specific pipeline step."""
    steps = {
        1: ('Data Cleaning', step1_data_cleaning.run),
        2: ('Windowing', step2_windowing.run),
        3: ('Feature Extraction', step3_feature_extraction.run),
        '3b': ('Advanced Features', step3b_advanced_features.run),
        '3c': ('Combine Features', step3c_combine_features.run),
        4: ('Scaling', step4_scaling.run),
        5: ('Feature Selection', step5_feature_selection.run),
        6: ('Model Training', step6_model_training.run),
        '6b': ('Optimized Models', step6b_optimized_models.run),
        '6c': ('Ensemble Models', step6c_ensemble_models.run),
        7: ('Results Summary', step7_results_summary.run),
        8: ('Spectral Features', step8_spectral_features.run),
        9: ('Spectral Scaling', step9_spectral_scaling.run),
        10: ('Spectral Selection', step10_spectral_selection.run),
        11: ('Spectral Model Training', step11_spectral_model_training.run),
        12: ('Final Comparison', step12_final_comparison.run)
    }
    
    # Handle both int and string step numbers
    try:
        step_key = int(step_num)
    except ValueError:
        step_key = step_num
    
    if step_key not in steps:
        print(f"Invalid step number: {step_num}")
        print("Valid steps: 1-12, 3b, 3c, 6b, 6c")
        return
    
    name, func = steps[step_key]
    logger.log(f"Running Step {step_num}: {name}")
    func()


def run_from_step(start_step):
    """Execute pipeline from a specific step onwards."""
    if start_step < 1 or start_step > 12:
        print(f"Invalid step number: {start_step}")
        return
    
    logger.header(f"RUNNING PIPELINE FROM STEP {start_step}")
    
    for step_num in range(start_step, 13):
        run_step(step_num)


def print_usage():
    """Print usage information."""
    print("""
WISDM-51 Activity Recognition Pipeline

Usage:
    python run_pipeline.py              # Run complete baseline pipeline
    python run_pipeline.py optimized    # Run optimized pipeline (80%+ target)
    python run_pipeline.py <step>       # Run specific step (1-12, 3b, 3c, 6b, 6c)
    python run_pipeline.py from <step>  # Run from step onwards

Steps (Part 1 - Time-Domain):
    1  - Data Cleaning
    2  - Windowing  
    3  - Feature Extraction
    3b - Advanced Features (wavelet, entropy, jerk)
    3c - Combine Features (basic + advanced + spectral)
    4  - Scaling
    5  - Feature Selection
    6  - Model Training (baseline)
    6b - Optimized Models (GridSearchCV)
    6c - Ensemble Models (Voting, Stacking)
    7  - Results Summary

Steps (Part 2 - Spectral):
    8  - Spectral Feature Extraction
    9  - Spectral Scaling
    10 - Spectral Feature Selection
    11 - Spectral Model Training
    12 - Final Comparison

Examples:
    python run_pipeline.py              # Full baseline pipeline (steps 1-12)
    python run_pipeline.py optimized    # Optimized pipeline for 80%+ accuracy
    python run_pipeline.py 4            # Run step 4 only
    python run_pipeline.py 6b           # Run optimized model training
    python run_pipeline.py from 8       # Run spectral pipeline (steps 8-12)
""")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_full_pipeline()
    elif len(sys.argv) == 2:
        if sys.argv[1] in ['-h', '--help']:
            print_usage()
        elif sys.argv[1] == 'optimized':
            run_optimized_pipeline()
        else:
            try:
                step_num = int(sys.argv[1])
                run_step(step_num)
            except ValueError:
                # Handle string steps like '3b', '6c'
                run_step(sys.argv[1])
    elif len(sys.argv) == 3 and sys.argv[1] == 'from':
        try:
            start_step = int(sys.argv[2])
            run_from_step(start_step)
        except ValueError:
            print_usage()
    else:
        print_usage()
