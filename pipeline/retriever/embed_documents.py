'''import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import ConfigConstants

def embed_documents(documents, embedding_path="embeddings.faiss"):
    embedding_model = HuggingFaceEmbeddings(model_name=ConfigConstants.EMBEDDING_MODEL_NAME)

    if os.path.exists(embedding_path):
        logging.info("Loading embeddings from local file")
        vector_store = FAISS.load_local(embedding_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        logging.info("Generating and saving embeddings")
        vector_store = FAISS.from_texts([doc['text'] for doc in documents], embedding_model)
        vector_store.save_local(embedding_path)
    
    return vector_store'''

import os
import logging
import hashlib
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import ConfigConstants  


def embed_documents(documents: List[Dict], embedding_path: str = ConfigConstants.DATA_SET_PATH + "embeddings/embeddings.faiss", metadata_path: str = ConfigConstants.DATA_SET_PATH + "embeddings/metadata.json") -> FAISS:
    logging.info(f"Total documents got :{len(documents)}")
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    embedding_model = HuggingFaceEmbeddings(model_name=ConfigConstants.EMBEDDING_MODEL_NAME)
    
    if os.path.exists(embedding_path) and os.path.exists(metadata_path):
        logging.info("Loading embeddings and metadata from local files")
        vector_store = FAISS.load_local(embedding_path, embedding_model, allow_dangerous_deserialization=True)
        existing_metadata = _load_metadata(metadata_path)
    else:
        # Initialize FAISS with at least one document to avoid the IndexError
        if documents:
            vector_store = FAISS.from_texts([documents[0]['text']], embedding_model)
        else:
            # If no documents are provided, initialize an empty FAISS index with a dummy document
            vector_store = FAISS.from_texts(["dummy document"], embedding_model)
        existing_metadata = {}
    
    # Identify new or modified documents
    new_documents = []
    for doc in documents:
        doc_hash = _generate_document_hash(doc['text'])
        if doc_hash not in existing_metadata:
            new_documents.append(doc)
            existing_metadata[doc_hash] = True  # Mark as processed
    
    if new_documents:
        logging.info(f"Generating embeddings for {len(new_documents)} new documents")
        with ThreadPoolExecutor() as executor:
            futures = []
            for doc in new_documents:
                futures.append(executor.submit(_embed_single_document, doc, embedding_model))
            
            for future in tqdm(futures, desc="Generating embeddings", unit="doc"):
                vector_store.add_texts([future.result()])
        
        # Save updated embeddings and metadata
        vector_store.save_local(embedding_path)
        _save_metadata(metadata_path, existing_metadata)
    else:
        logging.info("No new documents to process. Using existing embeddings.")
    
    return vector_store

def _embed_single_document(doc: Dict, embedding_model: HuggingFaceEmbeddings) -> str:
    return doc['text']

def _generate_document_hash(text: str) -> str:
    """Generate a unique hash for a document based on its text."""
    return hashlib.sha256(text.encode()).hexdigest()

def _load_metadata(metadata_path: str) -> Dict[str, bool]:
    """Load metadata from a file."""
    import json
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}

def _save_metadata(metadata_path: str, metadata: Dict[str, bool]):
    """Save metadata to a file."""
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


