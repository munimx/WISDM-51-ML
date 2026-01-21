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
import step6b_optimized_models
import step6c_ensemble_models
import step8_spectral_features


def run_full_pipeline():
    """Execute the complete pipeline - now redirects to optimized pipeline."""
    logger.log("Note: Legacy pipeline removed. Running optimized pipeline instead.")
    run_optimized_pipeline()


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
        '6b': ('Optimized Models', step6b_optimized_models.run),
        '6c': ('Ensemble Models', step6c_ensemble_models.run),
        8: ('Spectral Features', step8_spectral_features.run)
    }
    
    # Handle both int and string step numbers
    try:
        step_key = int(step_num)
    except ValueError:
        step_key = step_num
    
    if step_key not in steps:
        print(f"Invalid step number: {step_num}")
        print("Valid steps: 1-5, 8, 3b, 3c, 6b, 6c")
        return
    
    name, func = steps[step_key]
    logger.log(f"Running Step {step_num}: {name}")
    func()


def run_from_step(start_step):
    """Execute pipeline from a specific step onwards - deprecated."""
    logger.log("Note: Sequential step execution deprecated. Use 'python run_pipeline.py optimized' instead.")
    logger.log(f"To run individual steps, use: python run_pipeline.py step {start_step}")


def print_usage():
    """Print usage information."""
    print("""
WISDM-51 Activity Recognition Pipeline

Usage:
    python run_pipeline.py              # Run complete baseline pipeline
    python run_pipeline.py optimized    # Run optimized pipeline (80%+ target)
    python run_pipeline.py step <num>   # Run specific step (1-5, 8, 3b, 3c, 6b, 6c)

Steps (Optimized Pipeline):
    1  - Data Cleaning
    2  - Windowing  
    3  - Feature Extraction
    4  - Scaling
    5  - Feature Selection
    6b - Optimized Models (RandomSearch with early stopping)
    6c - Ensemble Models (Voting, Stacking)
    8  - Spectral Feature Extraction

Examples:
    python run_pipeline.py optimized    # Full optimized pipeline (~3h 22min)
    python run_pipeline.py step 6b      # Run optimized model training only
    python run_pipeline.py step 6c      # Run ensemble models only
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
