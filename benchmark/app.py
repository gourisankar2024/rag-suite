import gradio as gr
import os
import json
import pandas as pd
from scripts.evaluate_negative_rejection import evaluate_negative_rejection
from scripts.helper import update_config
from scripts.evaluate_noise_robustness import evaluate_noise_robustness
from scripts.evaluate_factual_robustness import evaluate_factual_robustness

# Path to score files
Noise_Robustness_DIR = "results/Noise Robustness/"
Negative_Rejection_DIR = "results/Negative Rejection/"
Counterfactual_Robustness_DIR = "results/Counterfactual Robustness/"

# Function to read and aggregate score data
def load_noise_robustness_scores():
    models = set()
    noise_rates = set()
    
    if not os.path.exists(Noise_Robustness_DIR):
        return pd.DataFrame(columns=["Noise Ratio"])

    score_data = {}

    # Read all JSON score files
    for filename in os.listdir(Noise_Robustness_DIR):
        if filename.startswith("scores_") and filename.endswith(".json"):
            filepath = os.path.join(Noise_Robustness_DIR, filename)
            with open(filepath, "r") as f:
                score = json.load(f)
                model = score["model"]
                noise_rate = str(score["noise_rate"])

                models.add(model)
                noise_rates.add(noise_rate)

                score_data[(model, noise_rate)] = score["accuracy"]

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "Noise Ratio": model,
            **{
                rate: f"{score_data.get((model, rate), 'N/A') * 100:.2f}%" 
                if score_data.get((model, rate), "N/A") != "N/A" 
                else "N/A"
                for rate in sorted(noise_rates, key=float)
            }
        }
        for model in sorted(models)
    ])

    return df

# Function to load Negative Rejection scores (Only for Noise Rate = 1.0)
def load_negative_rejection_scores():
    if not os.path.exists(Negative_Rejection_DIR):
        return pd.DataFrame()

    score_data = {}
    models = set()

    for filename in os.listdir(Negative_Rejection_DIR):
        if filename.startswith("scores_") and filename.endswith(".json") and "_noise_1.0_" in filename:
            filepath = os.path.join(Negative_Rejection_DIR, filename)
            with open(filepath, "r") as f:
                score = json.load(f)
                model = filename.split("_")[1]  # Extract model name
                models.add(model)
                score_data[model] = score.get("reject_rate", "N/A")

    df = pd.DataFrame([
        {"Model": model, "Rejection Rate": f"{score_data.get(model, 'N/A') * 100:.2f}%"
         if score_data.get(model, "N/A") != "N/A"
         else "N/A"}
        for model in sorted(models)
    ])

    return df if not df.empty else pd.DataFrame(columns=["Model", "Rejection Rate"])

def load_counterfactual_robustness_scores():
    models = set()
    noise_rates = set()
    
    if not os.path.exists(Counterfactual_Robustness_DIR):
        return pd.DataFrame(columns=["Noise Ratio"])

    score_data = {}

    # Read all JSON score files
    for filename in os.listdir(Counterfactual_Robustness_DIR):
        if filename.startswith("scores_") and filename.endswith(".json"):
            filepath = os.path.join(Counterfactual_Robustness_DIR, filename)
            with open(filepath, "r") as f:
                score = json.load(f)
                model = filename.split("_")[1]
                noise_rate = str(score["noise_rate"])

                models.add(model)
                noise_rates.add(noise_rate)

                score_data[(model, noise_rate)] = score["reject_rate"]

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "Noise Ratio": model,
            **{
                rate: f"{score_data.get((model, rate), 'N/A') * 100:.2f}%" 
                if score_data.get((model, rate), "N/A") != "N/A" 
                else "N/A"
                for rate in sorted(noise_rates, key=float)
            }
        }
        for model in sorted(models)
    ])

    return df

# Gradio UI
def launch_gradio_app(dataset, config):
    with gr.Blocks() as app:
        gr.Markdown("# RAG System Evaluation on RGB Dataset #")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Noise Robustness")
                noise_table = gr.Dataframe(value=load_noise_robustness_scores(), interactive=False)

            with gr.Column():
                gr.Markdown("### Negative Rejection")
                rejection_table = gr.Dataframe(value=load_negative_rejection_scores(), interactive=False)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Counterfactual Robustness")
                counter_factual_table = gr.Dataframe(value=load_counterfactual_robustness_scores(), interactive=False)

            with gr.Column():
                gr.Markdown("### Information Integration")
                #rejection_table = gr.Dataframe(value=load_negative_rejection_scores(), interactive=False)

        def refresh_scores():
            return load_noise_robustness_scores(), load_negative_rejection_scores(), load_counterfactual_robustness_scores()

        with gr.Row():
            refresh_btn = gr.Button("Refresh", scale=0)
        refresh_btn.click(refresh_scores, outputs=[noise_table, rejection_table, counter_factual_table])

        # Inputs for Config Update
        with gr.Row():
            model_name_input = gr.Textbox(label="Model Name", value="llama3-8b-8192", interactive=True, scale=1)
            noise_rate_input = gr.Slider(label="Noise Rate", minimum=0, maximum=1.0, step=0.2, value=0.2, interactive=True, scale=1)
            num_queries_input = gr.Number(label="Number of Queries", value=50, interactive=True, scale=1)

        with gr.Row():
            recalculate_noise_robustness_btn = gr.Button("Noise Robustness", min_width=None)
            recalculate_negative_rejection_btn = gr.Button("Negative Rejection", min_width=None)
            recalculate_counter_factual_btn = gr.Button("Counter Factual", min_width=None)
            recalculate_counter_factual_btn = gr.Button("Integration Information", min_width=None)

        # Button to trigger noise robustness(config)
        def recalculate_noise_robustness(model_name, noise_rate, num_queries):
            # Update config with user-provided values
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_noise_robustness(dataset, config)
            return load_noise_robustness_scores()  # Reload scores after recalculating
        
        recalculate_noise_robustness_btn.click(
            recalculate_noise_robustness,
            inputs=[model_name_input, noise_rate_input, num_queries_input],
            outputs=[noise_table]
        )

        # Button to trigger evaluate_factual_robustness(config)
        def recalculate_counter_factual(model_name, noise_rate, num_queries):
            # Update config with user-provided values
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_factual_robustness(config)
            return load_counterfactual_robustness_scores()  # Reload scores after recalculating
        
        recalculate_counter_factual_btn.click(
            recalculate_counter_factual,
            inputs=[model_name_input, noise_rate_input, num_queries_input],
            outputs=[counter_factual_table]
        )
        
        # Button to trigger evaluate_factual_robustness(config)
        def recalculate_negative_rejection(model_name, noise_rate, num_queries):
            # Update config with user-provided values
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_negative_rejection(config)
            return load_negative_rejection_scores()  # Reload scores after recalculating
        
        recalculate_negative_rejection_btn.click(
            recalculate_negative_rejection,
            inputs=[model_name_input, noise_rate_input, num_queries_input],
            outputs=[rejection_table]
        )
    app.launch()
