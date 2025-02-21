import logging
from scripts.helper import adaptive_delay, load_dataset, load_used_data
from scripts.process_data import process_data
from scripts.groq_client import GroqClient
from scripts.prediction import predict

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get prediction from LLM based on different dataset

def get_prediction_result(config, data_file_name, prediction_file_name='', correct_rate = 0):
    results = []
    used_data = []
    dataset = load_dataset(data_file_name)
    modelname = config['model_name']
    num_queries = min(config['num_queries'], len(dataset))
    subdataset = dataset[:num_queries]

    # Create GroqClient instance for supported models
    if modelname in config['models']:
        model = GroqClient(plm=modelname)
    else:
        logging.warning(f"Skipping unknown model: {modelname}")
        return
    
    if config['UsePreCalculatedValue']: 
        logging.info(f"Trying to use pre calculated values for report generation")
        used_data = load_used_data(prediction_file_name)
    else:
        logging.info(f"Running evaluation for {num_queries} queries...")

    # Iterate through dataset and process queries
    for idx, instance in enumerate(subdataset, start=0):
        if instance['id'] in used_data and instance['query'] == used_data[instance['id']]['query'] and instance['answer']  == used_data[instance['id']]['ans']:
                results.append(used_data[instance['id']])
                continue
        
        logging.info(f"Executing Query {idx + 1} for Model: {modelname}")
        query, ans, docs = process_data(instance, config['noise_rate'], config['passage_num'], data_file_name, correct_rate)

        # Retry mechanism for prediction
        for attempt in range(1, config['retry_attempts'] + 1):
            label, prediction, factlabel = predict(query, ans, docs, model, "Document:\n{DOCS} \n\nQuestion:\n{QUERY}", 0.7)
            if prediction:  # If response is not empty, break retry loop
                break
            adaptive_delay(attempt)

        # Check correctness and log the result
        is_correct = all(x == 1 for x in label)  # True if all values are 1 (correct), else False
        logging.info(f"Model Response: {prediction}")
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
            'noise_rate': config['noise_rate'],
            'factlabel': factlabel
        }
        results.append(new_instance)

    return results
