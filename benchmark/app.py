import gradio as gr
from scripts.get_scores import load_counterfactual_robustness_scores, load_negative_rejection_scores, load_scores_common
from scripts.evaluate_information_integration import evaluate_information_integration
from scripts.evaluate_negative_rejection import evaluate_negative_rejection
from scripts.helper import initialize_logging, update_config, update_logs_periodically
from scripts.evaluate_noise_robustness import evaluate_noise_robustness
from scripts.evaluate_factual_robustness import evaluate_factual_robustness

Noise_Robustness_DIR = "results/Noise Robustness/"
Negative_Rejection_DIR = "results/Negative Rejection/"
Counterfactual_Robustness_DIR = "results/Counterfactual Robustness/"
Infomration_Integration_DIR = "results/Information Integration/"
 
# Gradio UI
def launch_gradio_app(config):
    initialize_logging()
    
    def toggle_switch(value):
        config['UsePreCalculatedValue'] = value

    with gr.Blocks() as app:
        app.title = "RAG System Evaluation"
        gr.Markdown("# RAG System Evaluation on RGB Dataset")

        # Add the description here
        gr.Markdown("""
        Welcome to the **RAG System Evaluation on RGB Dataset**! This tool is designed to evaluate and compare the performance of various **Large Language Models (LLMs)** using Retrieval-Augmented Generation (RAG) on the [**RGB dataset**](https://github.com/chen700564/RGB). The evaluation focuses on key metrics such as **Noise Robustness**, **Negative Rejection**, **Counterfactual Robustness**, and **Information Integration**. These metrics help assess how well different models handle noisy inputs, reject invalid queries, manage counterfactual scenarios, and integrate information effectively.

        #### Key Features:
        - **Compare Multiple LLMs**: Evaluate and compare the performance of different LLMs side by side.
        - **Pre-calculated Metrics**: View results from pre-computed evaluations for quick insights.
        - **Recalculate Metrics**: Option to recalculate metrics for custom configurations.
        - **Interactive Controls**: Adjust model parameters, noise rates, and query counts to explore model behavior under different conditions.
        - **Detailed Reports**: Visualize results in clear, interactive tables for each evaluation metric.

        #### How to Use:
        1. **Select a Model**: Choose from the available LLMs to evaluate.
        2. **Configure Model Settings**: Adjust the noise rate and set the number of queries.
        3. **Choose Evaluation Mode**: Use pre-calculated values for quick results or recalculate metrics for custom analysis.
        4. **Compare Results**: Review and compare the evaluation metrics across different models in the tables below.
        5. **Logs**: View live logs to monitor what's happening behind the scenes in real-time.
                    
        """)

        # Top Section - Inputs and Controls
        with gr.Accordion("Model Settings", open=True):
            with gr.Row():
                with gr.Column():
                    model_name_input = gr.Dropdown(
                        label="Model Name",
                        choices=config['models'],
                        value=config['models'][0],
                        interactive=True
                    )
                    noise_rate_input = gr.Slider(
                        label="Noise Rate",
                        minimum=0,
                        maximum=1.0,
                        step=0.2,
                        value=config['noise_rate'],
                        interactive=True
                    )
                    num_queries_input = gr.Number(
                        label="Number of Queries",
                        value=config['num_queries'],
                        interactive=True
                    )
                with gr.Column():
                    toggle = gr.Checkbox(
                        label="Use pre-calculated values?",
                        value=True,
                        info="If checked, the report(s) will use pre-calculated metrics from saved output files. If any report has N/A value, Click on respective report generation button to generate value based on configuration. Uncheck to recalculate the metrics again."
                    )
                    refresh_btn = gr.Button("Refresh", variant="primary", scale= 0)
        # Next Section - Action Buttons
        with gr.Accordion("Evaluation Actions", open=True):
            with gr.Row():
                recalculate_noise_btn = gr.Button("Evaluate Noise Robustness")
                recalculate_negative_btn = gr.Button("Evaluate Negative Rejection")
                recalculate_counterfactual_btn = gr.Button("Evaluate Counterfactual Robustness")
                recalculate_integration_btn = gr.Button("Evaluate Integration Information")

        # Middle Section - Data Tables
        with gr.Accordion("Evaluation Results", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Noise Robustness\n**Description:** The experimental result of noise robustness measured by accuracy (%) under different noise ratios.")
                    noise_table = gr.Dataframe(value=load_scores_common(Noise_Robustness_DIR, config), interactive=False)
                with gr.Column():
                    gr.Markdown("### 🚫 Negative Rejection\n**Description:** This measures the model's ability to reject invalid or nonsensical queries.")
                    rejection_table = gr.Dataframe(value=load_negative_rejection_scores(config), interactive=False)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔄 Counterfactual Robustness\n**Description:** Evaluates a model's ability to handle errors in external knowledge.")
                    counter_factual_table = gr.Dataframe(value=load_counterfactual_robustness_scores(config), interactive=False)
                with gr.Column():
                    gr.Markdown("### 🧠 Information Integration\n**Description:** The experimental result of information integration measured by accuracy (%) under different noise ratios.")
                    integration_table = gr.Dataframe(value=load_scores_common(Infomration_Integration_DIR, config), interactive=False)

        # Logs Section
        with gr.Accordion("View Live Logs", open=False):
            log_section = gr.Textbox(label="Logs", interactive=False, lines=10, every=2)

        # Event Handling
        toggle.change(toggle_switch, inputs=toggle)
        app.queue() 
        app.load(update_logs_periodically, outputs=log_section)
        # Refresh Scores Function
        def refresh_scores(model_name, noise_rate, num_queries):
            update_config(config, model_name, noise_rate, num_queries)
            return load_scores_common(Noise_Robustness_DIR, config), load_negative_rejection_scores(config), load_counterfactual_robustness_scores(config), load_scores_common(Infomration_Integration_DIR, config)

        refresh_btn.click(refresh_scores, inputs=[model_name_input, noise_rate_input, num_queries_input], outputs=[noise_table, rejection_table, counter_factual_table, integration_table])

        # Button Functions
        def recalculate_noise_robustness(model_name, noise_rate, num_queries):
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_noise_robustness(config)
            return load_scores_common(Noise_Robustness_DIR, config)
        
        recalculate_noise_btn.click(recalculate_noise_robustness, inputs=[model_name_input, noise_rate_input, num_queries_input], outputs=[noise_table])

        def recalculate_counterfactual_robustness(model_name, noise_rate, num_queries):
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_factual_robustness(config)
            return load_counterfactual_robustness_scores(config)
        
        recalculate_counterfactual_btn.click(recalculate_counterfactual_robustness, inputs=[model_name_input, noise_rate_input, num_queries_input], outputs=[counter_factual_table])

        def recalculate_negative_rejection(model_name, noise_rate, num_queries):
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_negative_rejection(config)
            return load_negative_rejection_scores(config)
        
        recalculate_negative_btn.click(recalculate_negative_rejection, inputs=[model_name_input, noise_rate_input, num_queries_input], outputs=[rejection_table])

        def recalculate_integration_info(model_name, noise_rate, num_queries):
            update_config(config, model_name, noise_rate, num_queries)
            evaluate_information_integration(config)
            return load_scores_common(Infomration_Integration_DIR, config)
        
        recalculate_integration_btn.click(recalculate_integration_info , inputs=[model_name_input, noise_rate_input, num_queries_input], outputs=[integration_table])

    app.launch()
