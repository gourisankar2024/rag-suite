import json
import os
import time
import logging
from pathlib import Path

# Create a list to store logs
logs = []

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

def load_config(config_file="config.json"):
    """Load configuration from the config file."""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            config['output_file_extension'] = f'{config['model_name']}_noise_{config['noise_rate']}_passage_{config['passage_num']}_num_queries_{config['num_queries']}'
        return config
    except Exception as e:
        logging.info(f"Error loading config: {e}")
        return {}
    
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
        config['model_name'] = model_name
    if noise_rate is not None:  # Explicitly check for None to handle 0.0
        config['noise_rate'] = float(noise_rate)  # Ensure it's a float
    if num_queries is not None:  # Explicitly check for None to handle 0
        config['num_queries'] = int(num_queries)  # Ensure it's an integer
    
    config['output_file_extension'] = f'{config['model_name']}_noise_{config['noise_rate']}_passage_{config['passage_num']}_num_queries_{config['num_queries']}'
    
    return config

def load_dataset(file_name):
    dataset = []
    with open('data/' + file_name, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line.strip()))  # Load each JSON object per line
    logging.info(f"Loaded {len(dataset)} entries from file {file_name}")  # Check how many records were loaded
    return dataset

def initialize_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Custom log handler to capture logs and add them to the logs list
    class LogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            logs.append(log_entry)

    # Add custom log handler to the logger
    log_handler = LogHandler()
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(log_handler)

def get_logs():
    """Retrieve logs for display."""
    return "\n".join(logs[-50:])

def load_used_data(filepath):
        """Loads existing processed data to avoid redundant evaluations."""
        used_data = {}
        if Path(filepath).exists():
            with open(filepath, encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    used_data[data['id']] = data
        return used_data


def update_logs_periodically():
    while True:
        time.sleep(2)  # Wait for 2 seconds
        yield get_logs() 
  