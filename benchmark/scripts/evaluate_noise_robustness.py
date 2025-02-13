import os
import json
import logging
from scripts.helper import adaptive_delay, ensure_directory_exists
from scripts.process_data import process_data
from scripts.groq_client import GroqClient
from scripts.prediction import predict

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Improved function to evaluate noise robustness
def evaluate_noise_robustness(dataset, config):
    result_path = config["result_path"] + 'Noise Robustness/'
    models = config["models"]
    noise_rate = config["noise_rate"]
    passage_num = config["passage_num"]
    num_queries = config["num_queries"]
    retry_attempts = config["retry_attempts"]


    # Iterate over each model specified in the config
    filename = os.path.join(result_path, f'prediction_{config["model_name"]}_noise_{noise_rate}_passage_{passage_num}.json')
    ensure_directory_exists(filename)
    useddata = {}

    # Load existing results if file exists
    '''if os.path.exists(filename):
        logging.info(f"Loading existing results from {filename}")
        with open(filename) as f:
            for line in f:
                data = json.loads(line)
                useddata[data['id']] = data'''

    results = []  # Store results for this model

    # Create GroqClient instance for supported models
    if config["model_name"] in models:
        model = GroqClient(plm=config["model_name"])
    else:
        logging.warning(f"Skipping unknown model: {config["model_name"]}")
        return
    
    # Iterate through dataset and process queries
    for idx, instance in enumerate(dataset[:num_queries], start=0):
        logging.info(f"Executing Query {idx + 1} for Model: {config["model_name"]}")
        query, ans, docs = process_data(instance, noise_rate, passage_num, "en_refine.json")

        # Retry mechanism for prediction
        for attempt in range(1, retry_attempts + 1):
            label, prediction, factlabel = predict(query, ans, docs, model, "Document:\n{DOCS} \n\nQuestion:\n{QUERY}", 0.7, instance)
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
            'noise_rate': noise_rate,
            'factlabel': factlabel
        }
        results.append(new_instance)

    # Save results to a file
    with open(filename, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Compute per-model noise robustness
    correct_count = sum(1 for res in results if 0 not in res['label'] and 1 in res['label'])
    accuracy = correct_count / len(results) if results else 0

    # Calculate tt and all_rate metrics
    tt = sum(1 for i in results if (noise_rate == 1 and i['label'][0] == -1) or (0 not in i['label'] and 1 in i['label']))
    all_rate = tt / len(results) if results else 0

    # Save the final score file with tt and all_rate
    scores = {
        'model': config["model_name"],
        'accuracy': accuracy,
        'noise_rate': noise_rate,
        'correct_count': correct_count,
        'total': len(results),
        'all_rate': all_rate,
        'tt': tt
    }
    logging.info(f"score: {scores}")
    logging.info(f"Noise Robustness Accuracy: {accuracy:.2%}")
    
    score_filename = os.path.join(result_path, f'scores_{config["model_name"]}_noise_{noise_rate}_passage_{passage_num}.json')
    with open(score_filename, 'w') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return results
