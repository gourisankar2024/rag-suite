
import os

class ConfigConstants:
    # Constants related to datasets and models
    DATA_SET_PATH= '/persistent/'
    DATA_SET_NAMES = ['covidqa', 'cuad', 'techqa','delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa']
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    RE_RANKER_MODEL_NAME = 'cross-encoder/ms-marco-electra-base'
    GENERATION_MODEL_NAME = 'mixtral-8x7b-32768'
    VALIDATION_MODEL_NAME = 'llama3-70b-8192'
    GENERATION_MODELS = ["llama3-8b-8192", "qwen-2.5-32b", "mixtral-8x7b-32768", "gemma2-9b-it" ]
    VALIDATION_MODELS = ["llama3-70b-8192", "deepseek-r1-distill-llama-70b" ]
    DEFAULT_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

class AppConfig:
    def __init__(self, vector_store, gen_llm, val_llm):
        self.vector_store = vector_store
        self.gen_llm = gen_llm
        self.val_llm = val_llm
        self.loaded_datasets = self.detect_loaded_datasets()  # Auto-detect loaded datasets

    @staticmethod
    def detect_loaded_datasets():
        print('Calling detect_loaded_datasets')
        """Check which datasets are already stored locally."""
        local_path = ConfigConstants.DATA_SET_PATH + 'local_datasets'
        if not os.path.exists(local_path):
            return set()
        
        dataset_files = os.listdir(local_path)
        loaded_datasets = {
            file.replace("_test.pkl", "") for file in dataset_files if file.endswith("_test.pkl")
        }
        return loaded_datasets