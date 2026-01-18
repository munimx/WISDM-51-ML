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
import step4_scaling
import step5_feature_selection
import step6_model_training
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


def run_step(step_num):
    """Execute a specific pipeline step."""
    steps = {
        1: ('Data Cleaning', step1_data_cleaning.run),
        2: ('Windowing', step2_windowing.run),
        3: ('Feature Extraction', step3_feature_extraction.run),
        4: ('Scaling', step4_scaling.run),
        5: ('Feature Selection', step5_feature_selection.run),
        6: ('Model Training', step6_model_training.run),
        7: ('Results Summary', step7_results_summary.run),
        8: ('Spectral Features', step8_spectral_features.run),
        9: ('Spectral Scaling', step9_spectral_scaling.run),
        10: ('Spectral Selection', step10_spectral_selection.run),
        11: ('Spectral Model Training', step11_spectral_model_training.run),
        12: ('Final Comparison', step12_final_comparison.run)
    }
    
    if step_num not in steps:
        print(f"Invalid step number: {step_num}")
        print("Valid steps: 1-12")
        return
    
    name, func = steps[step_num]
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
    python run_pipeline.py              # Run complete pipeline
    python run_pipeline.py <step>       # Run specific step (1-12)
    python run_pipeline.py from <step>  # Run from step onwards

Steps (Part 1 - Time-Domain):
    1 - Data Cleaning
    2 - Windowing  
    3 - Feature Extraction
    4 - Scaling
    5 - Feature Selection
    6 - Model Training
    7 - Results Summary

Steps (Part 2 - Spectral):
    8 - Spectral Feature Extraction
    9 - Spectral Scaling
    10 - Spectral Feature Selection
    11 - Spectral Model Training
    12 - Final Comparison

Examples:
    python run_pipeline.py          # Full pipeline (steps 1-12)
    python run_pipeline.py 4        # Run step 4 only
    python run_pipeline.py from 8   # Run spectral pipeline (steps 8-12)
""")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_full_pipeline()
    elif len(sys.argv) == 2:
        if sys.argv[1] in ['-h', '--help']:
            print_usage()
        else:
            try:
                step_num = int(sys.argv[1])
                run_step(step_num)
            except ValueError:
                print_usage()
    elif len(sys.argv) == 3 and sys.argv[1] == 'from':
        try:
            start_step = int(sys.argv[2])
            run_from_step(start_step)
        except ValueError:
            print_usage()
    else:
        print_usage()
