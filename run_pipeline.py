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


def run_full_pipeline():
    """Execute the complete pipeline from start to finish."""
    start_time = time.time()
    
    logger.header("WISDM-51 ACTIVITY RECOGNITION PIPELINE")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("")
    
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
        7: ('Results Summary', step7_results_summary.run)
    }
    
    if step_num not in steps:
        print(f"Invalid step number: {step_num}")
        print("Valid steps: 1-7")
        return
    
    name, func = steps[step_num]
    logger.log(f"Running Step {step_num}: {name}")
    func()


def run_from_step(start_step):
    """Execute pipeline from a specific step onwards."""
    if start_step < 1 or start_step > 7:
        print(f"Invalid step number: {start_step}")
        return
    
    logger.header(f"RUNNING PIPELINE FROM STEP {start_step}")
    
    for step_num in range(start_step, 8):
        run_step(step_num)


def print_usage():
    """Print usage information."""
    print("""
WISDM-51 Activity Recognition Pipeline

Usage:
    python run_pipeline.py              # Run complete pipeline
    python run_pipeline.py <step>       # Run specific step (1-7)
    python run_pipeline.py from <step>  # Run from step onwards

Steps:
    1 - Data Cleaning
    2 - Windowing  
    3 - Feature Extraction
    4 - Scaling
    5 - Feature Selection
    6 - Model Training
    7 - Results Summary

Examples:
    python run_pipeline.py          # Full pipeline
    python run_pipeline.py 4        # Run step 4 only
    python run_pipeline.py from 5   # Run steps 5, 6, 7
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
