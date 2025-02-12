import json
import tqdm
from pathlib import Path
import logging
from scripts.groq_client import GroqClient
from scripts.helper import adaptive_delay, ensure_directory_exists
from scripts.prompt import get_factual_prompt

def evaluate_factual_robustness(config):
    """Evaluates negative rejection for a given model by processing predictions and computing scores."""
    
    modelname = config["model_name"]
    noise_rate = config["noise_rate"]
    passage_num = config["passage_num"]
    
    if config["model_name"] in config["models"]:
        model = GroqClient(plm=config["model_name"])
    else:
        logging.warning(f"Skipping unknown model: {config["model_name"]}")
        return

    # File paths
    base_path = "results"
    evalue_file = f"{base_path}/Noise Robustness/prediction_{modelname}_noise_{noise_rate}_passage_{passage_num}.json"
    output_file = f"{base_path}/Counterfactual Robustness/output_{modelname}_noise_{noise_rate}_passage_{passage_num}.json"
    result_file = f"{base_path}/Counterfactual Robustness/scores_{modelname}_noise_{noise_rate}_passage_{passage_num}.json"
    ensure_directory_exists(output_file)
    
    def load_used_data(filepath):
        """Loads existing processed data to avoid redundant evaluations."""
        used_data = {}
        if Path(filepath).exists():
            with open(filepath, encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    used_data[data['id']] = data
        return used_data

    def process_query(model, data, used_data, output_file):
        """Processes a single query, generates evaluation, and writes the result."""
        if data['id'] in used_data and data['query'] == used_data[data['id']]['query'] and data['ans'] == used_data[data['id']]['ans']:
            output_file.write(json.dumps(used_data[data['id']], ensure_ascii=False) + '\n')
            return used_data[data['id']]

        try:
            instruction = get_factual_prompt(data['query'], data['prediction'])
            
            # Retry mechanism for evaluation
            for attempt in range(1, 4):
                evaluation = model.generate(instruction)
                if evaluation:
                    break
                adaptive_delay(attempt)
            
            data['evaluation'] = evaluation
            print(f"Model Response for Factual robustness: {evaluation}")
            output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            return data

        except Exception as e:
            print(f"Error processing query: {e}")
            return None

    def calculate_scores(results, config):
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
            'reject_rate': rejecttt/len(results),
            'all_rate': (tt)/len(results),
            'correct_rate': correct_tt/rejecttt if rejecttt > 0 else 0,
            'tt':tt,
            'rejecttt':rejecttt,
            'correct_tt':correct_tt,
            'nums': len(results),
            'noise_rate': config["noise_rate"],
        }
        return scores
    
    used_data = load_used_data(output_file)
    results = []

    with open(output_file, 'w', encoding='utf-8') as f_out, open(evalue_file, 'r', encoding='utf-8') as f_eval:
        for line in tqdm.tqdm(f_eval):
            data = json.loads(line)
            processed_data = process_query(model, data, used_data, f_out)
            if processed_data:
                results.append(processed_data)

    # Compute scores and save
    scores = calculate_scores(results, config)
    print(f"Score: {scores}")

    with open(result_file, 'w', encoding='utf-8') as f_result:
        json.dump(scores, f_result, ensure_ascii=False, indent=4)
