
class ConfigConstants:
    # Constants related to datasets and models
    DATA_SET_PATH= '/persistent/'
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    RE_RANKER_MODEL_NAME = 'cross-encoder/ms-marco-electra-base'
    GENERATION_MODEL_NAME = 'gemma2-9b-it'
    GENERATION_MODELS = ["llama3-8b-8192", "qwen-2.5-32b", "gemma2-9b-it" ]
    DEFAULT_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
