import logging
from scripts.helper import adaptive_delay, load_dataset
from scripts.process_data import process_data
from scripts.groq_client import GroqClient
from scripts.prediction import predict

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get prediction from LLM based on different dataset

def get_prediction_result(config, data_file_name):
    results = []
    dataset = load_dataset(data_file_name)
    # Create GroqClient instance for supported models
    if config["model_name"] in config["models"]:
        model = GroqClient(plm=config["model_name"])
    else:
        logging.warning(f"Skipping unknown model: {config["model_name"]}")
        return
    
    # Iterate through dataset and process queries
    for idx, instance in enumerate(dataset[:config["num_queries"]], start=0):
        logging.info(f"Executing Query {idx + 1} for Model: {config["model_name"]}")

        query, ans, docs = process_data(instance, config["noise_rate"], config["passage_num"], data_file_name)

        # Retry mechanism for prediction
        for attempt in range(1, config["retry_attempts"] + 1):
            label, prediction, factlabel = predict(query, ans, docs, model, "Document:\n{DOCS} \n\nQuestion:\n{QUERY}", 0.7)
            if prediction:  # If response is not empty, break retry loop
                break
            adaptive_delay(attempt)

        # Check correctness and log the result
        is_correct = all(x == 1 for x in label)  # True if all values are 1 (correct), else False
        #logging.info(f"Model Response: {prediction}")
        logging.info(f"Correctness: {is_correct}")

        # Save result for this query
        instance['label'] = label
        new_instance = {
            'id': instance['id'],
            'query': query,
            'ans': ans,
            'label': label,
            'prediction': prediction,
            'docs': docs,
            'noise_rate': config["noise_rate"],
            'factlabel': factlabel
        }
        results.append(new_instance)

    return results
