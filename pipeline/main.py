import logging
from config import AppConfig, ConfigConstants
from data.load_dataset import load_data
from generator.compute_rmse_auc_roc_metrics import compute_rmse_auc_roc_metrics
from retriever.chunk_documents import chunk_documents
from retriever.embed_documents import embed_documents
from generator.initialize_llm import initialize_generation_llm
from generator.initialize_llm import initialize_validation_llm
from app import launch_gradio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting the RAG pipeline")

    # Dictionary to store chunked documents
    all_chunked_documents = []
    datasets = {}

    # Load multiple datasets
    for data_set_name in ConfigConstants.DATA_SET_NAMES:
        logging.info(f"Loading dataset: {data_set_name}")
        datasets[data_set_name] = load_data(data_set_name)

        # Set chunk size based on dataset name
        chunk_size = ConfigConstants.DEFAULT_CHUNK_SIZE
        if data_set_name == 'cuad':
            chunk_size = 4000  # Custom chunk size for 'cuad'
        
        # Chunk documents
        chunked_documents = chunk_documents(datasets[data_set_name], chunk_size=chunk_size, chunk_overlap=ConfigConstants.CHUNK_OVERLAP)
        all_chunked_documents.extend(chunked_documents)  # Combine all chunks

    # Access individual datasets
    #for name, dataset in datasets.items():
        #logging.info(f"Loaded {name} with {dataset.num_rows} rows")
    
    # Logging final count
    logging.info(f"Total chunked documents: {len(all_chunked_documents)}")
    
    # Embed the documents
    vector_store = embed_documents(all_chunked_documents)
    logging.info("Documents embedded")
    
     # Initialize the Generation LLM
    gen_llm = initialize_generation_llm(ConfigConstants.GENERATION_MODEL_NAME)

    # Initialize the Validation LLM
    val_llm = initialize_validation_llm(ConfigConstants.VALIDATION_MODEL_NAME)

    #Compute RMSE and AUC-ROC for entire dataset
    #Enable below code for calculation
    #data_set_name = 'covidqa'
    #compute_rmse_auc_roc_metrics(gen_llm, val_llm, datasets[data_set_name], vector_store, 10)
    
    # Launch the Gradio app
    config = AppConfig(vector_store= vector_store, gen_llm = gen_llm, val_llm = val_llm)
    launch_gradio(config)

    logging.info("Finished!!!")

if __name__ == "__main__":
    main()