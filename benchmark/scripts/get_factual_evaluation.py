import os
import json
import logging
from scripts.get_prediction_result import get_prediction_result
from scripts.helper import ensure_directory_exists, load_dataset


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Improved function to evaluate noise robustness
def get_factual_evaluation(config):
    result_path = config['result_path'] + 'Counterfactual Robustness/'
    noise_rate = config['noise_rate']
    passage_num = config['passage_num']
    model_name = config['model_name']

    # Iterate over each model specified in the config
    filename = os.path.join(result_path, f'prediction_{model_name}_noise_{noise_rate}_passage_{passage_num}.json')
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

    results = get_prediction_result(config, config['factual_file_name'])  # Store results for this model

    # Save results to a file
    with open(filename, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Compute per-model noise robustness
    tt = sum(1 for i in results if (noise_rate == 1 and i['label'][0] == -1) or (0 not in i['label'] and 1 in i['label']))
    scores = {
    'all_rate': (tt)/len(results),
    'noise_rate': noise_rate,
    'tt':tt,
    'nums': len(results),
    }
    fact_tt = 0
    correct_tt = 0
    for i in results:
        if i['factlabel'] == 1:
            fact_tt += 1
            if 0 not in i['label']:
                correct_tt += 1
    fact_check_rate = fact_tt/len(results)
    if fact_tt > 0:
        correct_rate = correct_tt/fact_tt
    else:
        correct_rate = 0
    scores['fact_check_rate'] = fact_check_rate
    scores['correct_rate'] = correct_rate
    scores['fact_tt'] = fact_tt
    scores['correct_tt'] = correct_tt

    #logging.info(f"score: {scores}")
    score_filename = os.path.join(result_path, f'scores_{model_name}_noise_{noise_rate}_passage_{passage_num}.json')
    with open(score_filename, 'w') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return filename
