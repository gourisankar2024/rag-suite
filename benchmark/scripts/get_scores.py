     
import os
import json
import re
import pandas as pd

# Path to score files
Noise_Robustness_DIR = "results/Noise Robustness/"
Negative_Rejection_DIR = "results/Negative Rejection/"
Counterfactual_Robustness_DIR = "results/Counterfactual Robustness/"
Infomration_Integration_DIR = "results/Information Integration/"


# Function to read and aggregate score data
def load_scores_common(file_dir, config):
   
    if not os.path.exists(file_dir):
        return pd.DataFrame(columns=["Model", "0.2", "0.4", "0.6", "0.8", "1.0"])

    score_data = {}

    # Define fixed noise rates as columns
    fixed_noise_rates = ["0.2", "0.4", "0.6", "0.8", "1.0"]

    # Iterate over each model in config['models']
    for model in config["models"]:
        # Regex pattern to match files for the current model
        pattern = re.compile(
            rf"^scores_{re.escape(model)}_noise_(?P<noise_rate>[\d.]+)_"
            rf"passage_{re.escape(str(config['passage_num']))}_num_queries_{re.escape(str(config['num_queries']))}\.json$"
        )

        model_scores = {rate: "N/A" for rate in fixed_noise_rates}  # Initialize all as "N/A"

        # Search for matching files in directory
        for filename in os.listdir(file_dir):
            match = pattern.match(filename)
            if match:
                noise_rate = match.group("noise_rate")  # Extract noise rate from filename
                
                if noise_rate in fixed_noise_rates:  # Only consider predefined noise rates
                    filepath = os.path.join(file_dir, filename)
                    with open(filepath, "r") as f:
                        score = json.load(f)
                        accuracy = score.get("accuracy", "N/A")
                        model_scores[noise_rate] = f"{accuracy * 100:.2f}"  # Convert to percentage

        # Store in score_data
        score_data[model] = model_scores

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "Model": model,
            **score_data[model]
        }
        for model in config["models"]
    ])

    return df

# Function to load Negative Rejection scores (Only for Noise Rate = 1.0)
def load_negative_rejection_scores(config):
    if not os.path.exists(Negative_Rejection_DIR):
        return pd.DataFrame()

    if not os.path.exists(Negative_Rejection_DIR):
        return pd.DataFrame(columns=["Model", "Rejection Rate"])

    score_data = {}

    # Iterate over each model in config['models']
    for model in config["models"]:
        # Expected filename pattern for each model
        expected_filename = f"scores_{model}_noise_1.0_passage_{config['passage_num']}_num_queries_{config['num_queries']}.json"
        filepath = os.path.join(Negative_Rejection_DIR, expected_filename)

        # Check if file exists
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                score = json.load(f)
                reject_rate = score.get("reject_rate", "N/A")
                score_data[model] = f"{reject_rate * 100:.2f}%" if reject_rate != "N/A" else "N/A"
        else:
            score_data[model] = "N/A"

    # Convert to DataFrame
    df = pd.DataFrame([
        {"Model": model, "Rejection Rate": score_data[model]}
        for model in config["models"]
    ])

    return df

def load_counterfactual_robustness_scores(config):
    #hard code noise rate to 0.4
    config['noise_rate'] = 0.4
    if not os.path.exists(Counterfactual_Robustness_DIR):
        return pd.DataFrame(columns=["Model", "Accuracy with factual docs (%)", "Error Detection Rate", "Correction Rate (%)"])

    score_data = {}

    # Iterate over each model in config['models']
    for model in config["models"]:  
        # Expected filename pattern for each model
        expected_filename = f"scores_{model}_noise_{config['noise_rate']}_passage_{config['passage_num']}_num_queries_{config['num_queries']}.json"
        filepath = os.path.join(Counterfactual_Robustness_DIR, expected_filename)

        # Check if file exists
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                score = json.load(f)
                score_data[model] = {
                    "Accuracy with factual docs (%)": round(score.get("all_rate", 0) * 100, 2),  # No decimal
                    "Error Detection Rate (%)": round(score.get("reject_rate", 0) * 100, 2), 
                    "Correction Rate (%)": round(score.get("correct_rate", 0) * 100, 2)  # 2 decimal places
                }
        else:
            score_data[model] = {  # Populate with "N/A" if file not found
                "Accuracy with factual docs (%)": "N/A",
                "Error Detection Rate (%)": "N/A",
                "Correction Rate (%)": "N/A"
            }

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "Model": model,
            "Accuracy with factual docs (%)": f"{score_data[model]['Accuracy with factual docs (%)']:.2f}" if score_data[model]["Accuracy with factual docs (%)"] != "N/A" else "N/A",
            "Error Detection Rate": f"{score_data[model]['Error Detection Rate (%)']:.2f}" if score_data[model]["Error Detection Rate (%)"] != "N/A" else "N/A",
            "Correction Rate (%)": f"{score_data[model]['Correction Rate (%)']:.2f}" if score_data[model]["Correction Rate (%)"] != "N/A" else "N/A"
        }
        for model in config["models"]
    ])

    return df
