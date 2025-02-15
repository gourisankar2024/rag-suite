import json
import os
import tqdm
import logging
from scripts.evaluate_noise_robustness import evaluate_noise_robustness
from scripts.groq_client import GroqClient
from scripts.helper import adaptive_delay, load_used_data
from scripts.prompt import get_prompt

def evaluate_negative_rejection(config):
    """Evaluates negative rejection for a given model by processing predictions and computing scores."""
    if config.get('noise_rate', 1) != 1:
        logging.warning("Noise rate is not 1.0. Exiting evaluation.")
        return
    
    config['noise_rate'] = 1.0 # Noise rate should be 1.0 for negative rejection evaluation
    modelname = config['model_name']
    noise_rate = config['noise_rate']
    
    if modelname in config['models']:
        model = GroqClient(plm=modelname)
    else:
        logging.warning(f"Skipping unknown model: {modelname}")
        return

    # File paths
    base_path = "results"
    evalue_file = f"{base_path}/Noise Robustness/prediction_{config['output_file_extension']}.json"
    output_file = f"{base_path}/Negative Rejection/output_{config['output_file_extension']}.json"
    result_file = f"{base_path}/Negative Rejection/scores_{config['output_file_extension']}.json"
    
    if not os.path.exists(evalue_file):
        logging.info(f"Evaluation file does not exist for model{modelname} and noise rate {noise_rate}.")
        logging.info("Generating evaluation file")
        evaluate_noise_robustness(config)
    
    def process_query(model, data, used_data, output_file):
        """Processes a single query, generates evaluation, and writes the result."""
        if data['id'] in used_data and data['query'] == used_data[data['id']]['query'] and data['ans'] == used_data[data['id']]['ans']:
            output_file.write(json.dumps(used_data[data['id']], ensure_ascii=False) + '\n')
            return used_data[data['id']]

        try:
            instruction = get_prompt(data['query'], data['prediction'])
            
            # Retry mechanism for evaluation
            for attempt in range(1, 4):
                evaluation = model.generate(instruction)
                if evaluation:
                    break
                adaptive_delay(attempt)
            
            data['evaluation'] = evaluation
            print(f"Model Response: {evaluation}")
            output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            return data

        except Exception as e:
            print(f"Error processing query: {e}")
            return None

    def calculate_scores(results):
        """Calculates and returns rejection rates and other metrics."""
        reject_count = sum(1 for i in results if "not addressed" in i['evaluation'])
        true_positive_count = sum(1 for i in results if 0 not in i['label'] and 1 in i['label'])
        total = len(results)

        return {
            'reject_rate': reject_count / total if total else 0,
            'all_rate': true_positive_count / total if total else 0,
            'tt': true_positive_count,
            'rejecttt': reject_count,
            'nums': total,
        }

    used_data = []
    results = []
    if config['UsePreCalculatedValue']: 
        logging.info(f"Trying to use pre calculated values for Negative rejection report generation")
        used_data = load_used_data(output_file)
    else:
        logging.info(f"Recalculating the metrics...")

    with open(output_file, 'w', encoding='utf-8') as f_out, open(evalue_file, 'r', encoding='utf-8') as f_eval:
        for line in tqdm.tqdm(f_eval):
            data = json.loads(line)
            processed_data = process_query(model, data, used_data, f_out)
            if processed_data:
                results.append(processed_data)

    # Compute scores and save
    scores = calculate_scores(results)
    logging.info(f"Negative Rejection Score: {scores}")

    with open(result_file, 'w', encoding='utf-8') as f_result:
        json.dump(scores, f_result, ensure_ascii=False, indent=4)
