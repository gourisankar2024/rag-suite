import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

def chunk_documents(page_list, doc_id, chunk_size=1000, chunk_overlap=200):
    """
    Chunk a list of page contents into smaller segments with document ID metadata.
    
    Args:
        page_list (list): List of strings, each string being the content of a page.
        doc_id (str): Unique identifier for the document.
        chunk_size (int): Maximum size of each chunk (default: 1000 characters).
        chunk_overlap (int): Overlap between chunks (default: 200 characters).
    
    Returns:
        list: List of dictionaries, each containing 'text', 'source', and 'doc_id'.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    seen_hashes = set()  # Track hashes of chunks to avoid duplicates

    for page_num, page_content in enumerate(page_list, start=1):  # Start page numbering at 1
        if not page_content or not isinstance(page_content, str):
            continue  # Skip empty or invalid pages

        # Split the page content into chunks
        chunks = text_splitter.split_text(page_content)
        
        for i, chunk in enumerate(chunks):
            # Generate a unique hash for the chunk
            chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
            
            # Skip if the chunk is a duplicate
            if chunk_hash in seen_hashes:
                continue
            
            # Create source identifier (e.g., "doc_123_page_1_chunk_0")
            source = f"doc_{doc_id}_page_{page_num}_chunk_{i}"
            
            # Add the chunk with doc_id as metadata
            documents.append({
                'text': chunk,
                'source': source,
                'doc_id': doc_id
            })
            seen_hashes.add(chunk_hash)
            
    logging.info(f"Chunking of documents is done. Chunked the document to {len(documents)} numbers of chunks")
    return documents