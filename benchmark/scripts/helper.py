import os
import time
import logging

# Helper function to ensure directory exists
def ensure_directory_exists(filepath):
    """Ensure the directory for a given file path exists."""
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Helper function for adaptive delay
def adaptive_delay(attempt, max_delay=60):
    """Increase wait time with each retry."""
    delay = min(5 * attempt, max_delay)  # Max delay of max_delay seconds
    logging.info(f"Retrying after {delay} seconds...")
    time.sleep(delay)