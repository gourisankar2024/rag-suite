import os
import logging
import pickle
from datasets import load_dataset
from config import ConfigConstants  # For saving the dataset locally

def load_data(data_set_name, local_path=ConfigConstants.DATA_SET_PATH):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    dataset_file = os.path.join(local_path, f"{data_set_name}_test.pkl")
    
    if os.path.exists(dataset_file):
        logging.info("Loading dataset {data_set_name} from local storage")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        logging.info("Loading dataset from Hugging Face")
        dataset = load_dataset("rungalileo/ragbench", data_set_name, split="test")
        logging.info(f"Saving {data_set_name} dataset locally")
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)
    
    logging.info("Dataset loaded successfully")
    logging.info(f"Number of documents found: {dataset.num_rows}")
    return dataset

