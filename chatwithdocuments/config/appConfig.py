import logging
from retriever.llm_manager import LLMManager
from retriever.document_manager import DocumentManager
from retriever.chat_manager import ChatManager

class AppConfig:
    def __init__(self):
        # Initialize LLMManager with the default model
        self.gen_llm = LLMManager()  # This will initialize the default model ("gemma2-9b-it")
        # Initialize DocumentManager (it will be a singleton instance shared across the app)
        self.doc_manager = DocumentManager()
        self.chat_manager = ChatManager(documentManager = self.doc_manager, llmManager = self.gen_llm)
        logging.info("AppConfig initialized with LLMManager")