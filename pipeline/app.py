import gradio as gr
import logging
import threading
import time
from generator.compute_metrics import get_attributes_text
from generator.generate_metrics import generate_metrics, retrieve_and_generate_response
from config import AppConfig, ConfigConstants
from generator.initialize_llm import initialize_generation_llm, initialize_validation_llm
from generator.document_utils import get_logs, initialize_logging 

def launch_gradio(config : AppConfig):
    """
    Launch the Gradio app with pre-initialized objects.
    """
    initialize_logging()

    def update_logs_periodically():
        while True:
            time.sleep(2)  # Wait for 2 seconds
            yield get_logs() 

    def answer_question(query, state):
        try:
            # Generate response using the passed objects
            response, source_docs = retrieve_and_generate_response(config.gen_llm, config.vector_store, query)
            
            # Update state with the response and source documents
            state["query"] = query
            state["response"] = response
            state["source_docs"] = source_docs
            
            response_text = f"Response: {response}\n\n"
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

            metrics_text = "Metrics:\n"
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
        """Generate and return the updated model information string."""
        return (
            f"Embedding Model: {ConfigConstants.EMBEDDING_MODEL_NAME}\n"
            f"Generation LLM: {config.gen_llm.name if hasattr(config.gen_llm, 'name') else 'Unknown'}\n"
            f"Validation LLM: {config.val_llm.name if hasattr(config.val_llm, 'name') else 'Unknown'}\n"
        )

    # Wrappers for event listeners
    def reinitialize_gen_llm(gen_llm_name):
        return reinitialize_llm("generation", gen_llm_name)

    def reinitialize_val_llm(val_llm_name):
        return reinitialize_llm("validation", val_llm_name)
    
    # Define Gradio Blocks layout
    with gr.Blocks() as interface:
        interface.title = "Real Time RAG Pipeline Q&A"
        gr.Markdown("# Real Time RAG Pipeline Q&A")  # Heading
        
        # Textbox for new generation LLM name
        with gr.Row():
            new_gen_llm_input = gr.Dropdown(
                label="Generation Model",
                choices=ConfigConstants.GENERATION_MODELS,  # Directly use the list
                value=ConfigConstants.GENERATION_MODELS[0] if ConfigConstants.GENERATION_MODELS else None,  # First value dynamically
                interactive=True
            )

            new_val_llm_input = gr.Dropdown(
                label="Validation Model",
                choices=ConfigConstants.VALIDATION_MODELS,  # Directly use the list
                value=ConfigConstants.VALIDATION_MODELS[0] if ConfigConstants.VALIDATION_MODELS else None,  # First value dynamically
                interactive=True
            )

            model_info_display = gr.Textbox(
                value=get_updated_model_info(),  # Use the helper function
                label="System Information",
                interactive=False  # Read-only textbox
            )

        # State to store response and source documents
        state = gr.State(value={"query": "","response": "", "source_docs": {}})
        gr.Markdown("Ask a question and get a response with metrics calculated from the RAG pipeline.")  # Description
        with gr.Row():
            query_input = gr.Textbox(label="Ask a question", placeholder="Type your query here")
        with gr.Row():
            submit_button = gr.Button("Submit", variant="primary", scale = 0)  # Submit button
            clear_query_button = gr.Button("Clear", scale = 0)  # Clear button
        with gr.Row():
            answer_output = gr.Textbox(label="Response", placeholder="Response will appear here")
        
        with gr.Row():
            compute_metrics_button = gr.Button("Compute metrics", variant="primary" , scale = 0)
            attr_output = gr.Textbox(label="Attributes", placeholder="Attributes will appear here")
            metrics_output = gr.Textbox(label="Metrics", placeholder="Metrics will appear here")
       
        #with gr.Row():
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
        with gr.Row():
            log_section = gr.Textbox(label="Logs", interactive=False, visible=True, lines=10 , every=2)  # Log section

        # Update UI when logs_state changes
        interface.queue() 
        interface.load(update_logs_periodically, outputs=log_section)

    interface.launch()