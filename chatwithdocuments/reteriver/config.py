

class ConfigConstants:
    # Constants related to datasets and models
    DATA_SET_PATH= '/persistent/'
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    RE_RANKER_MODEL_NAME = 'cross-encoder/ms-marco-electra-base'
    GENERATION_MODEL_NAME = 'mixtral-8x7b-32768'
    GENERATION_MODELS = ["llama3-8b-8192", "qwen-2.5-32b", "mixtral-8x7b-32768", "gemma2-9b-it" ]
    DEFAULT_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

class AppConfig:
    def __init__(self, vector_store, gen_llm, val_llm):
        self.vector_store = vector_store
        self.gen_llm = gen_llm
