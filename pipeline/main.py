import logging
from config import AppConfig, ConfigConstants
from generator.compute_rmse_auc_roc_metrics import compute_rmse_auc_roc_metrics
from retriever.load_selected_datasets import load_selected_datasets
from generator.initialize_llm import initialize_generation_llm
from generator.initialize_llm import initialize_validation_llm
from app import launch_gradio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting the RAG pipeline")

    # Initialize the Generation LLM
    gen_llm = initialize_generation_llm(ConfigConstants.GENERATION_MODEL_NAME)

    # Initialize the Validation LLM
    val_llm = initialize_validation_llm(ConfigConstants.VALIDATION_MODEL_NAME)

    #Compute RMSE and AUC-ROC for entire dataset
    #Enable below code for calculation
    #data_set_name = 'covidqa'
    #compute_rmse_auc_roc_metrics(gen_llm, val_llm, datasets[data_set_name], vector_store, 10)
    
    # Launch the Gradio app
    config = AppConfig(vector_store = None, gen_llm = gen_llm, val_llm = val_llm)
    load_selected_datasets(['covidqa'], config)
    launch_gradio(config)

    logging.info("Finished!!!")

if __name__ == "__main__":
    main()