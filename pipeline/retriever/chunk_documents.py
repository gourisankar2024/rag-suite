from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

def chunk_documents(dataset, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    seen_hashes = set()  # Track hashes of chunks to avoid duplicates

    for data in dataset:
        text_list = data['documents']
        for text in text_list:
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                # Generate a unique hash for the chunk
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
                
                # Skip if the chunk is a duplicate
                if chunk_hash in seen_hashes:
                    continue
                
                # Add the chunk to the documents list and track its hash
                documents.append({'text': chunk, 'source': f"{data['question']}_chunk_{i}"})
                seen_hashes.add(chunk_hash)
    
    return documents