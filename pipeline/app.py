import gradio as gr
import logging
import threading
import time
from generator.compute_metrics import get_attributes_text
from generator.generate_metrics import generate_metrics, retrieve_and_generate_response
from config import AppConfig, ConfigConstants
from generator.initialize_llm import initialize_generation_llm, initialize_validation_llm 

def launch_gradio(config : AppConfig):
    """
    Launch the Gradio app with pre-initialized objects.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a list to store logs
    logs = []

    # Custom log handler to capture logs and add them to the logs list
    class LogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            logs.append(log_entry)

    # Add custom log handler to the logger
    log_handler = LogHandler()
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(log_handler)

    def log_updater():
        """Background function to add logs."""
        while True:
            time.sleep(2)  # Update logs every 2 seconds
            pass  # Log capture is now handled by the logging system

    def get_logs():
        """Retrieve logs for display."""
        return "\n".join(logs[-50:])  # Only show the last 50 logs for example

    # Start the logging thread
    threading.Thread(target=log_updater, daemon=True).start()

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

    def reinitialize_gen_llm(gen_llm_name):
        """Reinitialize the generation LLM and return updated model info."""
        if gen_llm_name.strip():  # Only update if input is not empty
            config.gen_llm = initialize_generation_llm(gen_llm_name)
        
        # Return updated model information
        updated_model_info = (
            f"Embedding Model: {ConfigConstants.EMBEDDING_MODEL_NAME}\n"
            f"Generation LLM: {config.gen_llm.name if hasattr(config.gen_llm, 'name') else 'Unknown'}\n"
            f"Validation LLM: {config.val_llm.name if hasattr(config.val_llm, 'name') else 'Unknown'}\n"
        )
        return updated_model_info

    def reinitialize_val_llm(val_llm_name):
        """Reinitialize the generation LLM and return updated model info."""
        if val_llm_name.strip():  # Only update if input is not empty
            config.val_llm = initialize_validation_llm(val_llm_name)
        
        # Return updated model information
        updated_model_info = (
            f"Embedding Model: {ConfigConstants.EMBEDDING_MODEL_NAME}\n"
            f"Generation LLM: {config.gen_llm.name if hasattr(config.gen_llm, 'name') else 'Unknown'}\n"
            f"Validation LLM: {config.val_llm.name if hasattr(config.val_llm, 'name') else 'Unknown'}\n"
        )
        return updated_model_info
    
    # Define Gradio Blocks layout
    with gr.Blocks() as interface:
        interface.title = "Real Time RAG Pipeline Q&A"
        gr.Markdown("### Real Time RAG Pipeline Q&A")  # Heading
        
        # Textbox for new generation LLM name
        with gr.Row():
            new_gen_llm_input = gr.Textbox(label="New Generation LLM Name", placeholder="Enter LLM name to update")
            update_gen_llm_button = gr.Button("Update Generation LLM")
            new_val_llm_input = gr.Textbox(label="New Validation LLM Name", placeholder="Enter LLM name to update")
            update_val_llm_button = gr.Button("Update Validation LLM")

        # Section to display LLM names
        with gr.Row():
            model_info = f"Embedding Model: {ConfigConstants.EMBEDDING_MODEL_NAME}\n"
            model_info += f"Generation LLM: {config.gen_llm.name if hasattr(config.gen_llm, 'name') else 'Unknown'}\n"
            model_info += f"Validation LLM: {config.val_llm.name if hasattr(config.val_llm, 'name') else 'Unknown'}\n"
            model_info_display = gr.Textbox(value=model_info, label="Model Information", interactive=False)  # Read-only textbox

        # State to store response and source documents
        state = gr.State(value={"query": "","response": "", "source_docs": {}})
        gr.Markdown("Ask a question and get a response with metrics calculated from the RAG pipeline.")  # Description
        with gr.Row():
            query_input = gr.Textbox(label="Ask a question", placeholder="Type your query here")
        with gr.Row():
            submit_button = gr.Button("Submit", variant="primary")  # Submit button
            clear_query_button = gr.Button("Clear")  # Clear button
        with gr.Row():
            answer_output = gr.Textbox(label="Response", placeholder="Response will appear here")
        
        with gr.Row():
            compute_metrics_button = gr.Button("Compute metrics", variant="primary")
            attr_output = gr.Textbox(label="Attributes", placeholder="Attributes will appear here")
            metrics_output = gr.Textbox(label="Metrics", placeholder="Metrics will appear here")
       
        #with gr.Row():

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
        
        update_gen_llm_button.click(
            fn=reinitialize_gen_llm,
            inputs=[new_gen_llm_input],
            outputs=[model_info_display]  # Update the displayed model info
        )

        update_val_llm_button.click(
            fn=reinitialize_val_llm,
            inputs=[new_val_llm_input],
            outputs=[model_info_display]  # Update the displayed model info
        )
        
        # Section to display logs
        with gr.Row():
            start_log_button = gr.Button("Start Log Update", elem_id="start_btn")  # Button to start log updates
        with gr.Row():
            log_section = gr.Textbox(label="Logs", interactive=False, visible=True, lines=10)  # Log section

        # Set button click to trigger log updates
        start_log_button.click(fn=get_logs, outputs=log_section)

    interface.launch()