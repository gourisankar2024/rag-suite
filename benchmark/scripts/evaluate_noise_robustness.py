import os
import json
import logging
from scripts.get_prediction_result import get_prediction_result
from scripts.helper import ensure_directory_exists


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Improved function to evaluate noise robustness
def evaluate_noise_robustness(config):
    result_path = config["result_path"] + 'Noise Robustness/'
    noise_rate = config['noise_rate']
    passage_num = config['passage_num']

    # Iterate over each model specified in the config
    filename = os.path.join(result_path, f'prediction_{config['model_name']}_noise_{noise_rate}_passage_{passage_num}.json')
    ensure_directory_exists(filename)

    # Load existing results if file exists
    '''
    useddata = {}
    if os.path.exists(filename):
        logging.info(f"Loading existing results from {filename}")
        with open(filename) as f:
            for line in f:
                data = json.loads(line)
                useddata[data['id']] = data'''

    results = get_prediction_result(config, config["robustness_file_name"])  # Store results for this model

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
        'model': config['model_name'],
        'accuracy': accuracy,
        'noise_rate': noise_rate,
        'correct_count': correct_count,
        'total': len(results),
        'all_rate': all_rate,
        'tt': tt
    }
    logging.info(f"score: {scores}")
    logging.info(f"Noise Robustness Accuracy: {accuracy:.2%}")
    
    score_filename = os.path.join(result_path, f'scores_{config['model_name']}_noise_{noise_rate}_passage_{passage_num}.json')
    with open(score_filename, 'w') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return results
