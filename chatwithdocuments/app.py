import logging
import gradio as gr
from utils.document_utils import initialize_logging
from retriever.chat_manager import chat_response
 # Note: DocumentManager is now initialized in retrieve_documents.py
from globals import app_config 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
initialize_logging()

def load_sample_question(question):
    return question

def clear_selection():
    return [], "", []  # Reset doc_selector to empty list

def process_uploaded_file(file, current_selection):
    """Process uploaded file using DocumentManager and update UI."""
    status, page_list, filename, _ = app_config.doc_manager.process_document(file.name if file else None)

    # Update current selection to include new file if not already present
    updated_selection = current_selection if current_selection else []
    if filename and filename not in updated_selection:
        updated_selection.append(filename)

    return (
        status,
        page_list,
        gr.update(choices=app_config.doc_manager.get_uploaded_documents(), value=updated_selection)
    )

def update_doc_selector(selected_docs):
    """Keep selected documents in sync."""
    return selected_docs

# UI Configuration
models = ["qwen-2.5-32b", "gemma2-9b-it"]
example_questions = [
    "What is communication server?",
    "Show me an example of a configuration file.",
    "How to create Protected File Directories ?",
    "What are the attributes Azureblobstorage?",
    "What is Mediator help?",
    "Why AzureBlobStorage port is used?"
]
all_questions = [
    "Can you explain Communication Server architecture?",
    "Why does the other instance of my multi-instance qmgr seem to hang after a failover? Queue manager will not start after failover.",
    "Explain the concept of blockchain.",
    "What is the capital of France?",
    "Do Surface Porosity and Pore Size Influence Mechanical Properties and Cellular Response to PEEK?",
    "How does a vaccine work?",
    "Tell me the step-by-step instruction for front-door installation.",
    "What are the risk factors for heart disease?",
]

with gr.Blocks() as interface:
    interface.title = "🤖 IntelliDoc: AI Document Explorer"
    gr.Markdown("""
        # 🤖 IntelliDoc: AI Document Explorer
        **AI Document Explorer** allows you to upload PDF documents and interact with them using AI-powered analysis and summarization. Ask questions, extract key insights, and gain a deeper understanding of your documents effortlessly.
    """)
    with gr.Row():
        # Left Sidebar
        with gr.Column(scale=2):
            gr.Markdown("## Upload and Select Document")
            upload_btn = gr.File(label="Upload PDF Document", file_types=[".pdf"])
            doc_selector = gr.Dropdown(
                choices=app_config.doc_manager.get_uploaded_documents(),
                label="Documents",
                multiselect=True,
                value=[]  # Initial value as empty list
            )
            model_selector = gr.Dropdown(choices=models, label="Models", interactive=True)
            clear_btn = gr.Button("Clear Selection")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)

            # Process uploaded file and update UI
            upload_btn.change(
                process_uploaded_file,
                inputs=[upload_btn, doc_selector],
                outputs=[
                    upload_status,
                    gr.State(),  # page_list
                    doc_selector  # Update choices and value together
                ]
            )
            clear_btn.click(
                clear_selection,
                outputs=[doc_selector, upload_status, gr.State()]
            )
            # Reinitialize LLM when the model changes
            model_selector.change(
                app_config.gen_llm.reinitialize_llm,
                inputs=[model_selector],
                outputs=[upload_status]
            )

        # Middle Section (Chat & LLM Response)
        with gr.Column(scale=6):
            gr.Markdown("## Chat with document(s)")
            chat_history = gr.Textbox(label="Chat History", interactive=False, lines=28, elem_id="chat-history", elem_classes=["chat-box"])
            with gr.Row():
                chat_input = gr.Textbox(label="Ask additional questions about the document...", show_label=False, placeholder="Ask additional questions about the document...", elem_id="chat-input", lines=3)
                chat_btn = gr.Button("🚀 Send", variant="primary", elem_id="send-button", scale=0)
            chat_btn.click(chat_response, inputs=[chat_input, doc_selector, chat_history], outputs=chat_history).then(
                lambda: "",  # Return an empty string to clear the chat_input
                outputs=chat_input
            )

        # Right Sidebar (Sample Questions & History)
        with gr.Column(scale=2):
            gr.Markdown("## Frequently asked questions:")
            with gr.Column():
                gr.Examples(
                    examples=example_questions,
                    inputs=chat_input,
                    label=""
                )
                '''question_dropdown = gr.Dropdown(
                    label="",
                    choices=all_questions,
                    interactive=True,
                    info="Choose a question from the dropdown to populate the query box."
                )'''

            #gr.Markdown("## Logs")
            #history = gr.Textbox(label="Previous Queries", interactive=False)

    gr.HTML("""
    <style>
    .chat-box textarea {
        max-height: 600px !important;
        overflow-y: auto !important;
        resize: vertical;
        white-space: pre-wrap;  /* Keeps formatting */
    }
    </style>
    """)

if __name__ == "__main__":
    interface.launch()