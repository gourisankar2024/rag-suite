import logging
from data.load_dataset import load_data
from retriever.embed_documents import embed_documents
from retriever.chunk_documents import chunk_documents

loaded_datasets = set()  # Keep track of loaded datasets

def load_selected_datasets(selected_datasets, config) -> str:
    """Load, chunk, and embed selected datasets."""
    global loaded_datasets

    if not selected_datasets:
        return "No dataset selected."

    all_chunked_documents = []
    datasets = {}

    for data_set_name in selected_datasets:
        logging.info(f"Loading dataset: {data_set_name}")
        datasets[data_set_name] = load_data(data_set_name)

        # Set chunk size
        chunk_size = 4000 if data_set_name == 'cuad' else 1000  # Example chunk sizes
        
        # Chunk documents
        chunked_documents = chunk_documents(datasets[data_set_name], chunk_size=chunk_size, chunk_overlap=200)
        all_chunked_documents.extend(chunked_documents)
        # Logging final count
        logging.info(f"Total chunked documents: {len(all_chunked_documents)}")

        # Mark dataset as loaded
        loaded_datasets.add(data_set_name)

    # Embed documents
    config.vector_store = embed_documents(all_chunked_documents)
    logging.info("Documents embeding completed.")
    
    # **🔹 Refresh loaded datasets after loading**
    config.loaded_datasets = config.detect_loaded_datasets()

    return loaded_datasets #f"Loaded datasets: {', '.join(loaded_datasets)}"