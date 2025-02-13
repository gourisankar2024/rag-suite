import json
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

def update_config(config, model_name=None, noise_rate=None, num_queries=None):
    """
    Update the config dictionary with user-provided values.

    Args:
        config (dict): The configuration dictionary to update.
        model_name (str, optional): The model name to update in the config.
        noise_rate (float, optional): The noise rate to update in the config.
        num_queries (int, optional): The number of queries to update in the config.

    Returns:
        dict: The updated configuration dictionary.
    """
    if model_name:
        config["model_name"] = model_name
    if noise_rate is not None:  # Explicitly check for None to handle 0.0
        config["noise_rate"] = float(noise_rate)  # Ensure it's a float
    if num_queries is not None:  # Explicitly check for None to handle 0
        config["num_queries"] = int(num_queries)  # Ensure it's an integer
    return config

def load_dataset(file_name):
    dataset = []
    with open('data/' + file_name, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line.strip()))  # Load each JSON object per line
    logging.info(f"Loaded {len(dataset)} entries from file {file_name}")  # Check how many records were loaded
    return dataset