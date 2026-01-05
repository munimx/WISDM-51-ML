"""
WISDM-51 Activity Recognition Pipeline - Logger Utility
Provides logging functionality for all pipeline steps.
"""

import os
from datetime import datetime
from config import LOG_DIR


class Logger:
    def __init__(self, log_filename='pipeline_log.txt'):
        self.log_path = os.path.join(LOG_DIR, log_filename)
        os.makedirs(LOG_DIR, exist_ok=True)
    
    def log(self, message, print_msg=True):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        
        if print_msg:
            print(log_line)
        
        with open(self.log_path, 'a') as f:
            f.write(log_line + '\n')
    
    def header(self, title):
        """Log a section header."""
        self.log(title)
        self.log("-" * 40)
    
    def separator(self):
        """Log a separator line."""
        self.log("=" * 50)


# Global logger instance
logger = Logger()
