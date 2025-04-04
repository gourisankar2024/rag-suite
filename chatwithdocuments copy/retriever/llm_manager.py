import logging
import os
from typing import List, Dict, Any, Tuple
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class LLMManager:
    DEFAULT_MODEL = "gemma2-9b-it"  # Set the default model name

    def __init__(self):
        self.generation_llm = None
        logging.info("LLMManager initialized")

        # Initialize the default model during construction
        try:
            self.initialize_generation_llm(self.DEFAULT_MODEL)
            logging.info(f"Initialized default LLM model: {self.DEFAULT_MODEL}")
        except ValueError as e:
            logging.error(f"Failed to initialize default LLM model: {str(e)}")

    def initialize_generation_llm(self, model_name: str) -> None:
        """
        Initialize the generation LLM using the Groq API.

        Args:
            model_name (str): The name of the model to use for generation.

        Raises:
            ValueError: If GROQ_API_KEY is not set.
        """
        api_key = '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' #os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please add it in your environment variables.")
        
        os.environ["GROQ_API_KEY"] = api_key
        self.generation_llm = ChatGroq(model=model_name, temperature=0.7)
        self.generation_llm.name = model_name
        logging.info(f"Generation LLM {model_name} initialized")

    def reinitialize_llm(self, model_name: str) -> str:
        """
        Reinitialize the LLM with a new model name.

        Args:
            model_name (str): The name of the new model to initialize.

        Returns:
            str: Status message indicating success or failure.
        """
        try:
            self.initialize_generation_llm(model_name)
            return f"LLM model changed to {model_name}"
        except ValueError as e:
            logging.error(f"Failed to reinitialize LLM with model {model_name}: {str(e)}")
            return f"Error: Failed to change LLM model: {str(e)}"

    def generate_response(self, question: str, relevant_docs: List[Dict[str, Any]]) -> Tuple[str, List[Document]]:
        """
        Generate a response using the generation LLM based on the question and relevant documents.

        Args:
            question (str): The user's query.
            relevant_docs (List[Dict[str, Any]]): List of relevant document chunks with text, metadata, and scores.

        Returns:
            Tuple[str, List[Document]]: The LLM's response and the source documents used.

        Raises:
            ValueError: If the generation LLM is not initialized.
            Exception: If there's an error during the QA chain invocation.
        """
        if not self.generation_llm:
            raise ValueError("Generation LLM is not initialized. Call initialize_generation_llm first.")

        # Convert the relevant documents into LangChain Document objects
        documents = [
            Document(page_content=doc['text'], metadata=doc['metadata'])
            for doc in relevant_docs
        ]

        # Create a proper retriever by subclassing BaseRetriever
        class SimpleRetriever(BaseRetriever):
            def __init__(self, docs: List[Document], **kwargs):
                super().__init__(**kwargs)  # Pass kwargs to BaseRetriever
                self._docs = docs  # Use a private attribute to store docs
                logging.debug(f"SimpleRetriever initialized with {len(docs)} documents")

            def _get_relevant_documents(self, query: str) -> List[Document]:
                logging.debug(f"SimpleRetriever._get_relevant_documents called with query: {query}")
                return self._docs

            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                logging.debug(f"SimpleRetriever._aget_relevant_documents called with query: {query}")
                return self._docs

        # Instantiate the retriever
        retriever = SimpleRetriever(docs=documents)

        # Create a retrieval-based question-answering chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.generation_llm,
            retriever=retriever,
            return_source_documents=True
        )

        try:
            result = qa_chain.invoke({"query": question})
            response = result['result']
            source_docs = result['source_documents']
            logging.info(f"Generated response for question: {question} : {response}")
            return response, source_docs
        except Exception as e:
            logging.error(f"Error during QA chain invocation: {str(e)}")
            raise e