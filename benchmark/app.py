import gradio as gr
import os
import json
import pandas as pd
from scripts.evaluate_information_integration import evaluate_information_integration
from scripts.evaluate_negative_rejection import evaluate_negative_rejection
from scripts.helper import update_config
from scripts.evaluate_noise_robustness import evaluate_noise_robustness
from scripts.evaluate_factual_robustness import evaluate_factual_robustness

# Path to score files
Noise_Robustness_DIR = "results/Noise Robustness/"
Negative_Rejection_DIR = "results/Negative Rejection/"
Counterfactual_Robustness_DIR = "results/Counterfactual Robustness/"
Infomration_Integration_DIR = "results/Information Integration/"

# Function to read and aggregate score data
def load_scores(file_dir):
    models = set()
    noise_rates = set()
    
    if not os.path.exists(file_dir):
        return pd.DataFrame(columns=["Noise Ratio"])

    score_data = {}

    # Read all JSON score files
    for filename in os.listdir(file_dir):
        if filename.startswith("scores_") and filename.endswith(".json"):
            filepath = os.path.join(file_dir, filename)
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
                rate: f"{score_data.get((model, rate), 'N/A') * 100:.2f}" 
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
                #noise_rate = str(score["noise_rate"])

                models.add(model)
                score_data[model] = {
                    "Accuracy (%)": int(score["all_rate"] * 100),  # No decimal
                    "Error Detection Rate": int(score["reject_rate"] * 10),  
                    "Correction Rate (%)": round(score["correct_rate"] * 100, 2)  # 2 decimal places
                }

    # Convert to DataFrame
    df = pd.DataFrame([
         {
            "Model": model,
            "Accuracy (%)": score_data.get(model, {}).get("Accuracy (%)", "N/A"),
            "Error Detection Rate": score_data.get(model, {}).get("Error Detection Rate", "N/A"),
            "Correction Rate (%)": f"{score_data.get(model, {}).get('Correction Rate (%)', 'N/A'):.2f}"
        }   
        for model in sorted(models)
    ])

    return df

# Gradio UI
def launch_gradio_app(config):
    with gr.Blocks() as app:
        app.title = "RAG System Evaluation"
        gr.Markdown("# RAG System Evaluation on RGB Dataset")

        # Top Section - Inputs and Controls
        with gr.Row():
            model_name_input = gr.Dropdown(
            label="Model Name",
            choices= config["models"],
            value="llama3-8b-8192",
            interactive=True
            )
            noise_rate_input = gr.Slider(label="Noise Rate", minimum=0, maximum=1.0, step=0.2, value=0.2, interactive=True)
            num_queries_input = gr.Number(label="Number of Queries", value=50, interactive=True)

        # Bottom Section - Action Buttons
        with gr.Row():
            recalculate_noise_btn = gr.Button("Evaluate Noise Robustness")
            recalculate_negative_btn = gr.Button("Evaluate Negative Rejection")
            recalculate_counterfactual_btn = gr.Button("Evaluate Counterfactual Robustness")
            recalculate_integration_btn = gr.Button("Evaluate Integration Information")

        with gr.Row():
            refresh_btn = gr.Button("Refresh", variant="primary", scale = 0)

        # Middle Section - Data Tables
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Noise Robustness\n**Description:** The experimental result of noise robustness measured by accuracy (%) under different noise ratios. Result show that the increasing noise rate poses a challenge for RAG in LLMs.")
                noise_table = gr.Dataframe(value=load_scores(Noise_Robustness_DIR), interactive=False)
            with gr.Column():
                gr.Markdown("### 🚫 Negative Rejection\n**Description:** This measures the model's ability to reject invalid or nonsensical queries instead of generating incorrect responses. A higher rejection rate means the model is better at filtering unreliable inputs.")
                rejection_table = gr.Dataframe(value=load_negative_rejection_scores(), interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                    ### 🔄 Counterfactual Robustness  
                    **Description:**  
                    Counterfactual Robustness evaluates a model's ability to handle **errors in external knowledge** while ensuring reliable responses.  

                    **Key Metrics in this Report:**  
                    - **Accuracy (%)** → Measures the accuracy (%) of LLMs with counterfactual documents.  
                    - **Error Detection Rate (%)** → Measures how often the model **rejects** incorrect or misleading queries instead of responding.  
                    - **Correct Rate (%)** → Measures how often the model provides accurate responses despite **potential misinformation**.  
                    """)
                counter_factual_table = gr.Dataframe(value=load_counterfactual_robustness_scores(), interactive=False)
            with gr.Column():
                gr.Markdown("### 🧠 Information Integration\n**Description:** The experimental result of information integration measured by accuracy (%) under different noise ratios. The result show that information integration poses a challenge for RAG in LLMs")
                integration_table = gr.Dataframe(value=load_scores(Infomration_Integration_DIR), interactive=False)
        

        # Refresh Scores Function
        def refresh_scores():
            return load_scores(Noise_Robustness_DIR), load_negative_rejection_scores(), load_counterfactual_robustness_scores(), load_scores(Infomration_Integration_DIR)

        refresh_btn.click(refresh_scores, outputs=[noise_table, rejection_table, counter_factual_table, integration_table])

        # Button Functions
        def recalculate_noise_robustness(model_name, noise_rate, num_queries):
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_noise_robustness(config)
            return load_scores(Noise_Robustness_DIR)
        
        recalculate_noise_btn.click(recalculate_noise_robustness, inputs=[model_name_input, noise_rate_input, num_queries_input], outputs=[noise_table])

        def recalculate_counterfactual_robustness(model_name, noise_rate, num_queries):
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_factual_robustness(config)
            return load_counterfactual_robustness_scores()
        
        recalculate_counterfactual_btn.click(recalculate_counterfactual_robustness, inputs=[model_name_input, noise_rate_input, num_queries_input], outputs=[counter_factual_table])

        def recalculate_negative_rejection(model_name, noise_rate, num_queries):
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_negative_rejection(config)
            return load_negative_rejection_scores()
        
        recalculate_negative_btn.click(recalculate_negative_rejection, inputs=[model_name_input, noise_rate_input, num_queries_input], outputs=[rejection_table])

        def recalculate_integration_info(model_name, noise_rate, num_queries):
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_information_integration(config)
            return load_scores(Infomration_Integration_DIR)
        
        recalculate_integration_btn.click(recalculate_integration_info , inputs=[model_name_input, noise_rate_input, num_queries_input], outputs=[integration_table])

    app.launch()
