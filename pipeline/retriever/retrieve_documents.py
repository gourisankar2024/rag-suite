import logging
import numpy as np
from transformers import pipeline

from config import ConfigConstants

def retrieve_top_k_documents(vector_store, query, top_k=5):
    documents = vector_store.similarity_search(query, k=top_k)
    logging.info(f"Top {top_k} documents reterived for query")

    #documents = rerank_documents(query, documents)
    
    return documents 

# Reranking: Cross-Encoder for refining top-k results
def rerank_documents(query, documents):
    """
    Re-rank documents using a cross-encoder model.

    Parameters:
        query (str): The user's query.
        documents (list): List of LangChain Document objects.
        reranker_model_name (str): Hugging Face model name for re-ranking.

    Returns:
        list: Re-ranked list of Document objects with updated scores.
    """
    # Initialize the cross-encoder model
    reranker = pipeline("text-classification", model=ConfigConstants.RE_RANKER_MODEL_NAME, top_k=1)

    # Pair the query with each document's text
    rerank_inputs = [{"text": query, "text_pair": doc.page_content} for doc in documents]

    # Get relevance scores for each query-document pair
    scores = reranker(rerank_inputs)

   # Attach the new scores to the documents
    for doc, score in zip(documents, scores):
        doc.metadata["rerank_score"] = score[0]['score']  # Access score from the first item in the list

    # Sort documents by the rerank_score in descending order
    documents = sorted(documents, key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
    logging.info("Re-ranked documents using a cross-encoder model")

    return documents


# Query Handling: Retrieve top-k candidates using FAISS with IVF index not used only for learning
def retrieve_top_k_documents_manual(vector_store, query, top_k=5):
    """
    Retrieve top-k documents using FAISS index and optionally rerank them.

    Parameters:
        vector_store (FAISS): The vector store containing the FAISS index and docstore.
        query (str): The user's query string.
        top_k (int): The number of top results to retrieve.
        reranker_model_name (str): The Hugging Face model name for cross-encoder reranking.

    Returns:
        list: Top-k retrieved and reranked documents.
    """
    # Encode the query into a dense vector
    embedding_model = vector_store.embedding_function
    query_vector = embedding_model.embed_query(query)  # Encode the query
    query_vector = np.array([query_vector]).astype('float32')
    
    # Search the FAISS index for top_k results
    distances, indices = vector_store.index.search(query_vector, top_k)

    # Retrieve documents from the docstore
    documents = []
    for idx in indices.flatten():
        if idx == -1:  # FAISS can return -1 for invalid indices
            continue
        doc_id = vector_store.index_to_docstore_id[idx]

        # Access the internal dictionary of InMemoryDocstore
        internal_docstore = getattr(vector_store.docstore, "_dict", None)
        if internal_docstore and doc_id in internal_docstore:  # Check if doc_id exists
            document = internal_docstore[doc_id]
            documents.append(document)

    # Rerank the documents 
    documents = rerank_documents(query, documents)
    
    return documents