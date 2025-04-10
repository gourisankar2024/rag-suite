import logging
import os
from typing import Any, Dict, List
import uuid
from data.document_loader import DocumentLoader
from data.pdf_reader import PDFReader
from retriever.chunk_documents import chunk_documents
from retriever.vector_store_manager import VectorStoreManager

class DocumentManager:
    def __init__(self):
        self.doc_loader = DocumentLoader()
        self.pdf_reader = PDFReader()
        self.vector_manager = VectorStoreManager()
        self.uploaded_documents = {}
        self.chunked_documents = {}
        self.document_ids = {}
        logging.info("DocumentManager initialized")

    def process_document(self, file):
        """
        Process an uploaded file: load, read PDF, chunk, and store in vector store.
        Returns: (status_message, page_list, filename, doc_id)
        """
        try:
            if file is None:
                return "No file uploaded", [], None, None

            logging.info(f"Processing file: {file}")

            # Load and validate file
            file_path = self.doc_loader.load_file(file)
            filename = os.path.basename(file_path)

            # Read PDF content
            page_list = self.pdf_reader.read_pdf(file_path)

            # Store the uploaded document
            self.uploaded_documents[filename] = file_path

            # Generate a unique document ID
            doc_id = str(uuid.uuid4())
            self.document_ids[filename] = doc_id

            # Chunk the pages
            chunks = chunk_documents(page_list, doc_id, chunk_size=2000, chunk_overlap=300)
            self.chunked_documents[filename] = chunks

            # Add chunks to vector store
            self.vector_manager.add_documents(chunks)

            return (
                f"Successfully loaded {filename} with {len(page_list)} pages",
                page_list,
                filename,
                doc_id
            )

        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            return f"Error: {str(e)}", [], None, None

    def get_uploaded_documents(self):
        """Return the list of uploaded document filenames."""
        return list(self.uploaded_documents.keys())

    def get_chunks(self, filename):
        """Return chunks for a given filename."""
        return self.chunked_documents.get(filename, [])

    def get_document_id(self, filename):
        """Return the document ID for a given filename."""
        return self.document_ids.get(filename, None)
    
    def retrieve_top_k(self, query: str, selected_docs: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top K chunks across the selected documents based on the user's query.

        Args:
            query (str): The user's query.
            selected_docs (List[str]): List of selected document filenames from the dropdown.
            k (int): Number of top results to return (default is 5).

        Returns:
            List[Dict[str, Any]]: List of top K chunks with their text, metadata, and scores.
        """
        if not selected_docs:
            logging.warning("No documents selected for retrieval")
            return []

        all_results = []
        for filename in selected_docs:
            doc_id = self.get_document_id(filename)
            if not doc_id:
                logging.warning(f"No document ID found for filename: {filename}")
                continue

            # Search for relevant chunks within this document
            results = self.vector_manager.search(query, doc_id, k=k)
            all_results.extend(results)

        # Sort all results by score in descending order and take the top K
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_k_results = all_results[:k]

        # Log the list of retrieved documents
        logging.info(f"Result from search :{all_results} ")
        logging.info(f"Retrieved top {k} documents:")
        for i, result in enumerate(top_k_results, 1):
            doc_id = result['metadata'].get('doc_id', 'Unknown')
            filename = next((name for name, d_id in self.document_ids.items() if d_id == doc_id), 'Unknown')
            logging.info(f"{i}. Filename: {filename}, Doc ID: {doc_id}, Score: {result['score']:.4f}, Text: {result['text'][:200]}...")

        return top_k_results