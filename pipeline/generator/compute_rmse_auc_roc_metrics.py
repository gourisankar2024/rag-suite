
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from generator.generate_metrics import generate_metrics, retrieve_and_generate_response
import logging

def compute_rmse_auc_roc_metrics(gen_llm, val_llm, dataset, vector_store, num_question):

     # Lists to accumulate ground truths and predictions for AUC-ROC computation
    all_ground_truth_relevance = []
    all_predicted_relevance = []

    all_ground_truth_utilization = []
    all_predicted_utilization = []

    all_ground_truth_adherence = []
    all_predicted_adherence = []

    # For each question in dataset get the metrics
    for i, document in enumerate(dataset):
        # Extract ground truth metrics from dataset
        ground_truth_relevance = dataset[i]['relevance_score']
        ground_truth_utilization = dataset[i]['utilization_score']
        ground_truth_adherence = 1 if dataset[i]['adherence_score'] else 0
        
        query = document['question']
        logging.info(f'Query number: {i + 1}')
        # Call the generate_metrics for each query
        response, source_docs = retrieve_and_generate_response(gen_llm, vector_store, query)
        attributes, metrics = generate_metrics(val_llm, response, source_docs, query, 25)
        
        # Extract predicted metrics (ensure these are continuous if possible)
        predicted_relevance = metrics.get('Context Relevance', 0) if metrics else 0
        predicted_utilization = metrics.get('Context Utilization', 0) if metrics else 0
        predicted_adherence = 1 if metrics.get('Adherence', False) else 0
        
        # === Handle Continuous Inputs for RMSE ===
        all_ground_truth_relevance.append(ground_truth_relevance)
        all_predicted_relevance.append(predicted_relevance)
        all_ground_truth_utilization.append(ground_truth_utilization)
        all_predicted_utilization.append(predicted_utilization)
        
        all_ground_truth_adherence.append(ground_truth_adherence)
        all_predicted_adherence.append(predicted_adherence)

        if i == num_question:
          break
    
    # === Compute RMSE & AUC-ROC for the Entire Dataset ===
    try:
        logging.info(f"All Ground Truth Relevance: {all_ground_truth_relevance}")
        logging.info(f"All Predicted Relevance: {all_predicted_relevance}")
        relevance_rmse = root_mean_squared_error(all_ground_truth_relevance, all_predicted_relevance)
    except ValueError:
        relevance_rmse = None

    try:
        logging.info(f"All Ground Truth Utilization: {all_ground_truth_utilization}")
        logging.info(f"All Predicted Utilization: {all_predicted_utilization}")
        utilization_rmse = root_mean_squared_error(all_ground_truth_utilization, all_predicted_utilization)
    except ValueError:
        utilization_rmse = None

    try:
        logging.info(f"All Ground Truth Adherence: {all_ground_truth_adherence}")
        logging.info(f"All Predicted Adherence: {all_predicted_adherence}")
        adherence_auc = roc_auc_score(all_ground_truth_adherence, all_predicted_adherence)
    except ValueError:
        adherence_auc = None   

    logging.info(f"Relevance RMSE score: {relevance_rmse}")
    logging.info(f"Utilization RMSE score: {utilization_rmse}")
    logging.info(f"Overall Adherence AUC-ROC: {adherence_auc}")

    return relevance_rmse, utilization_rmse, adherence_auc
