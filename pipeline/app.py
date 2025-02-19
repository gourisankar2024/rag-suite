import gradio as gr
import logging
import time
from generator.compute_metrics import get_attributes_text
from generator.generate_metrics import generate_metrics, retrieve_and_generate_response
from config import AppConfig, ConfigConstants
from generator.initialize_llm import initialize_generation_llm, initialize_validation_llm
from generator.document_utils import get_logs, initialize_logging
from retriever.load_selected_datasets import load_selected_datasets 

def launch_gradio(config : AppConfig):
    """
    Launch the Gradio app with pre-initialized objects.
    """
    initialize_logging()

    # **🔹 Always get the latest loaded datasets**
    config.detect_loaded_datasets()

    def update_logs_periodically():
        while True:
            time.sleep(2)  # Wait for 2 seconds
            yield get_logs() 

    def answer_question(query, state):
        try:
            # Ensure vector store is updated before use
            if config.vector_store is None:
                return "Please load a dataset first.", state
            
            # Generate response using the passed objects
            response, source_docs = retrieve_and_generate_response(config.gen_llm, config.vector_store, query)
            
            # Update state with the response and source documents
            state["query"] = query
            state["response"] = response
            state["source_docs"] = source_docs
            
            response_text = f"Response from Model : {response}\n\n"
            return response_text, state
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"An error occurred: {e}", state

    def compute_metrics(state):
        try:
            logging.info(f"Computing metrics")
            
            # Retrieve response and source documents from state
            response = state.get("response", "")
            source_docs = state.get("source_docs", {})
            query = state.get("query", "")

            # Generate metrics using the passed objects
            attributes, metrics = generate_metrics(config.val_llm, response, source_docs, query, 1)
            
            attributes_text = get_attributes_text(attributes)

            metrics_text = ""
            for key, value in metrics.items():
                if key != 'response':
                    metrics_text += f"{key}: {value}\n"
            
            return attributes_text, metrics_text
        except Exception as e:
            logging.error(f"Error computing metrics: {e}")
            return f"An error occurred: {e}", ""

    def reinitialize_llm(model_type, model_name):
        """Reinitialize the specified LLM (generation or validation) and return updated model info."""
        if model_name.strip():  # Only update if input is not empty
            if model_type == "generation":
                config.gen_llm = initialize_generation_llm(model_name)
            elif model_type == "validation":
                config.val_llm = initialize_validation_llm(model_name)

        return get_updated_model_info()

    def get_updated_model_info():
        loaded_datasets_str = ", ".join(config.loaded_datasets) if config.loaded_datasets else "None"
        """Generate and return the updated model information string."""
        return (
            f"Embedding Model: {ConfigConstants.EMBEDDING_MODEL_NAME}\n"
            f"Generation LLM: {config.gen_llm.name if hasattr(config.gen_llm, 'name') else 'Unknown'}\n"
            f"Re-ranking LLM: {ConfigConstants.RE_RANKER_MODEL_NAME}\n"
            f"Validation LLM: {config.val_llm.name if hasattr(config.val_llm, 'name') else 'Unknown'}\n"
            f"Loaded Datasets: {loaded_datasets_str}\n"
        )

    # Wrappers for event listeners
    def reinitialize_gen_llm(gen_llm_name):
        return reinitialize_llm("generation", gen_llm_name)

    def reinitialize_val_llm(val_llm_name):
        return reinitialize_llm("validation", val_llm_name)
    
    # Function to update query input when a question is selected from the dropdown
    def update_query_input(selected_question):
        return selected_question
        
    # Define Gradio Blocks layout
    with gr.Blocks() as interface:
        interface.title = "Real Time RAG Pipeline Q&A"
        gr.Markdown("""
            # Real Time RAG Pipeline Q&A
            The **Retrieval-Augmented Generation (RAG) Pipeline** combines retrieval-based and generative AI models to provide accurate and context-aware answers to your questions. 
            It retrieves relevant documents from a dataset (e.g., COVIDQA, TechQA, FinQA) and uses a generative model to synthesize a response. 
            Metrics are computed to evaluate the quality of the response and the retrieval process.
            """)
        # Model Configuration
        with gr.Accordion("System Information", open=False):
            with gr.Accordion("DataSet", open=False):
                with gr.Row():
                    dataset_selector = gr.CheckboxGroup(ConfigConstants.DATA_SET_NAMES, label="Select Datasets to Load")
                    load_button = gr.Button("Load", scale= 0)

            with gr.Row():
                # Column for Generation Model Dropdown
                with gr.Column(scale=1):
                    new_gen_llm_input = gr.Dropdown(
                        label="Generation Model",
                        choices=ConfigConstants.GENERATION_MODELS,
                        value=ConfigConstants.GENERATION_MODELS[0] if ConfigConstants.GENERATION_MODELS else None,
                        interactive=True,
                        info="Select the generative model for response generation."
                    )
                
                # Column for Validation Model Dropdown
                with gr.Column(scale=1):
                    new_val_llm_input = gr.Dropdown(
                        label="Validation Model",
                        choices=ConfigConstants.VALIDATION_MODELS,
                        value=ConfigConstants.VALIDATION_MODELS[0] if ConfigConstants.VALIDATION_MODELS else None,
                        interactive=True,
                        info="Select the model for validating the response quality."
                    )
                
                # Column for Model Information
                with gr.Column(scale=2):
                    model_info_display = gr.Textbox(
                        value=get_updated_model_info(),  # Use the helper function
                        label="Model Configuration",
                        interactive=False,  # Read-only textbox
                        lines=5 
                    )
        
        # Query Section
        gr.Markdown("Ask a question and get a response with metrics calculated from the RAG pipeline.")
        all_questions = [
            "When was the first case of COVID-19 identified?",
            "What are the ages of the patients in this study?",
            "Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy,  insolvency, etc.)?",
            "Explain the concept of blockchain.",
            "What is the capital of France?",
            "Do Surface Porosity and Pore Size Influence Mechanical Properties and Cellular Response to PEEK??",
            "How does a vaccine work?",
            "What is the difference between RNA and DNA?",
            "What are the risk factors for heart disease?",
            "What is the role of insulin in the body?",
            # Add more questions as needed
        ]

        # Subset of questions to display as examples
        example_questions = [
            "When was the first case of COVID-19 identified?",
            "What are the ages of the patients in this study?",
            "What is the Hepatitis C virus?",
            "Explain the concept of blockchain.",
            "What is the capital of France?"
        ]  
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Ask a question ",
                        placeholder="Type your query here or select from examples/dropdown",
                        lines=2
                    )
                with gr.Row():
                    submit_button = gr.Button("Submit", variant="primary", scale=0)
                    clear_query_button = gr.Button("Clear", scale=0)
            with gr.Column():
                gr.Examples(
                    examples=example_questions,  # Make sure the variable name matches
                    inputs=query_input,
                    label="Try these examples:"
                )
                question_dropdown = gr.Dropdown(
                    label="",
                    choices=all_questions,
                    interactive=True,
                    info="Choose a question from the dropdown to populate the query box."
                )
        
        # Attach event listener to dropdown
        question_dropdown.change(
            fn=update_query_input,
            inputs=question_dropdown,
            outputs=query_input
        )

        # Response and Metrics
        with gr.Row():
            answer_output = gr.Textbox(label="Response", placeholder="Response will appear here", lines=2)
        
        with gr.Row():
            compute_metrics_button = gr.Button("Compute metrics", variant="primary" , scale = 0)
            attr_output = gr.Textbox(label="Attributes", placeholder="Attributes will appear here")
            metrics_output = gr.Textbox(label="Metrics", placeholder="Metrics will appear here")
       
        # State to store response and source documents
        state = gr.State(value={"query": "","response": "", "source_docs": {}})

        # Pass config to update vector store
        load_button.click(lambda datasets: (load_selected_datasets(datasets, config), get_updated_model_info()), inputs=dataset_selector)
        # Attach event listeners to update model info on change
        new_gen_llm_input.change(reinitialize_gen_llm, inputs=new_gen_llm_input, outputs=model_info_display)
        new_val_llm_input.change(reinitialize_val_llm, inputs=new_val_llm_input, outputs=model_info_display)

        # Define button actions
        submit_button.click(
            fn=answer_question,
            inputs=[query_input, state],
            outputs=[answer_output, state]
        )
        clear_query_button.click(fn=lambda: "", outputs=[query_input])  # Clear query input
        compute_metrics_button.click(
            fn=compute_metrics,
            inputs=[state],
            outputs=[attr_output, metrics_output]
        )
        
        # Section to display logs
        with gr.Accordion("View Live Logs", open=False):
            with gr.Row():
                log_section = gr.Textbox(label="Logs", interactive=False, visible=True, lines=10 , every=2)  # Log section

        # Update UI when logs_state changes
        interface.queue() 
        interface.load(update_logs_periodically, outputs=log_section)

    interface.launch()