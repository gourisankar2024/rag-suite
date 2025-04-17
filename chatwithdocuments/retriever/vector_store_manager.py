import os
import logging
from config.config import ConfigConstants
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self, embedding_path="embeddings.faiss"):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_path (str): Path to save/load the FAISS index.
        """
        self.embedding_path = embedding_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=ConfigConstants.EMBEDDING_MODEL_NAME)
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize or load the FAISS vector store."""
        if os.path.exists(self.embedding_path):
            logging.info("Loading embeddings from local file")
            return FAISS.load_local(
                self.embedding_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            '''logging.info("Creating new vector store")
            # Return an empty vector store; it will be populated when documents are added
            return FAISS.from_texts(
                texts=[""],  # Dummy text to initialize
                embedding=self.embedding_model,
                metadatas=[{"source": "init", "doc_id": "init"}]
            )'''
            logging.info("Creating new vector store (unpopulated)")
            return None 

    def add_documents(self, documents):
        """
        Add new documents to the vector store and save it.
        
        Args:
            documents (list): List of dictionaries with 'text', 'source', and 'doc_id'.
        """
        if not documents:
            return

        texts = [doc['text'] for doc in documents]
        metadatas = [{'source': doc['source'], 'doc_id': doc['doc_id']} for doc in documents]

        logging.info("Adding new documents to vector store")
        
        if not self.vector_store:
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas
            )
        else:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)

        self.vector_store.save_local(self.embedding_path)
        logging.info(f"Vector store updated and saved to {self.embedding_path}")

    def search(self, query, doc_id, k=10):
        """
        Search the vector store for relevant chunks, filtered by doc_id.
        
        Args:
            query (str): The user's query.
            doc_id (str): The document ID to filter by.
            k (int): Number of results to return.
        
        Returns:
            list: List of relevant document chunks with metadata and scores.
        """
        if not self.vector_store:
            return []

        try:
            query = " ".join(query.lower().split())
            # Define a filter function to match doc_id
            filter_fn = lambda metadata: metadata['doc_id'] == doc_id
            
            # Perform similarity search with filter
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_fn
            )
            
            # Format results
            return [{'text': doc.page_content, 'metadata': doc.metadata, 'score': score} for doc, score in results]
        
        except Exception as e:
            logging.error(f"Error during vector store search: {str(e)}")
            return []