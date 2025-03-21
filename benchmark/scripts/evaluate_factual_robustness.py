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

    conditions = get_conditions()
    base_path = "results/Counterfactual Robustness"
    result_file = f"{base_path}/scores_{config['output_file_extension']}.json"
    final_scores = {"conditions": []}

    for condition in conditions:
        logging.info(f"\nEvaluating condition: {condition['label']} (correct_rate={condition['correct_rate']}, noise_rate={condition['noise_rate']})")
        config['noise_rate'] = condition['noise_rate']
        update_config(config)
       
        pred_file = get_prediction_file(config, condition['correct_rate'])
        output_file = f"{base_path}/output_{config['output_file_extension']}.json"
        ensure_directory_exists(output_file)
        
        logging.info(f"Factual pred file for {condition['label']}: {pred_file}")

        results = load_or_recalculate_data(config, output_file, pred_file, model)
        scores = calculate_scores(results, condition)
        final_scores["conditions"].append(scores)
        logging.info(f"Counterfactual Robustness Score for {condition['label']}: {scores}")

    save_final_scores(result_file, final_scores)

def get_conditions():
    """Returns the conditions to test."""
    return [
        {"correct_rate": 1.0, "noise_rate": 0.2, "label": "factual_only"},
        {"correct_rate": 0.0, "noise_rate": 0.4, "label": "counterfactual"}
    ]

def load_or_recalculate_data(config, output_file, pred_file, model):
    """Loads or recalculates data based on the configuration."""
    used_data = []
    results = []
    if config['UsePreCalculatedValue']:
        logging.info(f"Trying to use pre-calculated values")
        used_data = load_used_data(output_file)
    else:
        logging.info(f"Recalculating the metrics...")

    with open(output_file, 'w', encoding='utf-8') as f_out, open(pred_file, 'r', encoding='utf-8') as f_eval:
        for line in tqdm.tqdm(f_eval):
            data = json.loads(line)
            processed_data = process_query(model, data, used_data, f_out)
            if processed_data:
                results.append(processed_data)
    return results

def process_query(model, data, used_data, output_file):
    """Processes a single query, generates evaluation, and writes the result."""
    if data['id'] in used_data and data['query'] == used_data[data['id']]['query'] and data['ans'] == used_data[data['id']]['ans']:
        output_file.write(json.dumps(used_data[data['id']], ensure_ascii=False) + '\n')
        return used_data[data['id']]

    try:
        instruction = get_factual_prompt(data['query'], data['prediction'])
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
        'reject_rate': rejecttt / len(results) if len(results) > 0 else 0,
        'all_rate': tt / len(results) if len(results) > 0 else 0,
        'correct_rate': correct_tt / rejecttt if rejecttt > 0 else 0,
        'tt': tt,
        'rejecttt': rejecttt,
        'correct_tt': correct_tt,
        'nums': len(results),
        'noise_rate': condition['noise_rate'],
        'condition_label': condition['label']
    }
    return scores

def save_final_scores(result_file, final_scores):
    """Saves the final scores to a file."""
    with open(result_file, 'w', encoding='utf-8') as f_result:
        json.dump(final_scores, f_result, ensure_ascii=False, indent=4)
