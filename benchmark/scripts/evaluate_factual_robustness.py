import json
import tqdm
import logging
from scripts.get_prediction_file import get_prediction_file
from scripts.groq_client import GroqClient
from scripts.helper import adaptive_delay, ensure_directory_exists, load_used_data, update_config
from scripts.prompt import get_factual_prompt

def evaluate_factual_robustness(config):
    """Evaluates negative rejection for a given model under multiple correct_rate/noise_rate conditions."""
    model_name = config['model_name']
    
    if model_name in config['models']:
        model = GroqClient(plm=model_name)
    else:
        logging.warning(f"Skipping unknown model: {model_name}")
        return

    # Define the conditions to test
    conditions = [
        {"correct_rate": 1.0, "noise_rate": 0.2, "label": "factual_only"},  # factual documents with some noisy documents
        {"correct_rate": 0.0, "noise_rate": 0.4, "label": "counterfactual"}  # Counterfactual + noise
    ]

    base_path = "results/Counterfactual Robustness"
    result_file = f"{base_path}/scores_{config['output_file_extension']}.json"
    final_scores = {"conditions": []}

    def process_query(model, data, used_data, output_file):
        """Processes a single query, generates evaluation, and writes the result."""
        if data['id'] in used_data and data['query'] == used_data[data['id']]['query'] and data['ans'] == used_data[data['id']]['ans']:
            output_file.write(json.dumps(used_data[data['id']], ensure_ascii=False) + '\n')
            return used_data[data['id']]

        try:
            instruction = get_factual_prompt(data['query'], data['prediction'])
            #eval_model = GroqClient(plm='llama3-70b-8192')
            for attempt in range(1, 4):
                evaluation = model.generate(instruction)
                if evaluation:
                    break
                adaptive_delay(attempt)
            
            data['evaluation'] = evaluation
            logging.info(f"Model Response for Factual robustness: {evaluation}")
            output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            return data

        except Exception as e:
            print(f"Error processing query: {e}")
            return None

    def calculate_scores(results, condition):
        """Calculates and returns rejection rates and other metrics."""
        rejecttt = 0
        tt = 0
        correct_tt = 0
        for i in results:
            if "has identified" in i['evaluation'] or "Yes" in i['evaluation']:
                rejecttt += 1
                if 0 not in i['label'] and 1 in i['label']:
                    correct_tt += 1
            if 0 not in i['label'] and 1 in i['label']:
                tt += 1

        scores = {
            'reject_rate': rejecttt / len(results) if len(results) > 0 else 0, #Error Detection Rate (ED)
            'all_rate': tt / len(results) if len(results) > 0 else 0,
            'correct_rate': correct_tt / rejecttt if rejecttt > 0 else 0, #Error Correction Rate (CR)
            'tt': tt,
            'rejecttt': rejecttt,
            'correct_tt': correct_tt,
            'nums': len(results),
            'noise_rate': condition['noise_rate'],
            'condition_label': condition['label']
        }
        return scores
 
    for condition in conditions:
        logging.info(f"\nEvaluating condition: {condition['label']} (correct_rate={condition['correct_rate']}, noise_rate={condition['noise_rate']})")
        
        # Update config with current condition's noise_rate
        config['noise_rate'] = condition['noise_rate']
        #config['passage_num'] = 10
        update_config(config)
       
        # File paths with condition-specific suffixes
        pred_file = get_prediction_file(config, condition['correct_rate'])
        output_file = f"{base_path}/output_{config['output_file_extension']}.json"
       
        ensure_directory_exists(output_file)
        
        logging.info(f"Factual pred file for {condition['label']}: {pred_file}")

        # Load or recalculate data
        used_data = []
        results = []
        if config['UsePreCalculatedValue']:
            logging.info(f"Trying to use pre-calculated values for {condition['label']}")
            used_data = load_used_data(output_file)
        else:
            logging.info(f"Recalculating the metrics for {condition['label']}...")

        with open(output_file, 'w', encoding='utf-8') as f_out, open(pred_file, 'r', encoding='utf-8') as f_eval:
            for line in tqdm.tqdm(f_eval):
                data = json.loads(line)
                processed_data = process_query(model, data, used_data, f_out)
                if processed_data:
                    results.append(processed_data)

        # Compute and save scores
        scores = calculate_scores(results, condition)
        final_scores["conditions"].append(scores)
        logging.info(f"Counterfactual Robustness Score for {condition['label']}: {scores}")

    with open(result_file, 'w', encoding='utf-8') as f_result:
        json.dump(final_scores, f_result, ensure_ascii=False, indent=4)
