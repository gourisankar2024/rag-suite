import json
import logging
from app import launch_gradio_app
from scripts.evaluate_factual_robustness import evaluate_factual_robustness
from scripts.evaluate_negative_rejection import evaluate_negative_rejection
from scripts.evaluate_noise_robustness import evaluate_noise_robustness
from scripts.download_files import download_file, get_file_list

def load_config(config_file="config.json"):
    """Load configuration from the config file."""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.info(f"Error loading config: {e}")
        return {}

def main():
    # Load configuration
    config = load_config()

    logging.info(f"Model: {config["model_name"]}")
    logging.info(f"Noise Rate: {config["noise_rate"]}")
    logging.info(f"Passage Number: {config["passage_num"]}")
    logging.info(f"Number of Queries: {config["num_queries"]}")

    # Download files from the GitHub repository
    files = get_file_list()
    for file in files:
        download_file(file)

    # Load dataset from the local JSON file
    

    # Call evaluate_noise_robustness for each noise rate and model
    #evaluate_noise_robustness(config)
    #evaluate_negative_rejection(config)
    #evaluate_factual_robustness(config)
    launch_gradio_app(config)
    
if __name__ == "__main__":
    main()
