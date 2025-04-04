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
models = ["gemma2-9b-it", "llama-guard-3-8b", "qwen-2.5-32b", "mistral-saba-24b"]
example_questions = [
    "Can you give me summary of the document?",
    "Key insights found in the document?",
    "What are the risks outlined in the document?",
    "Explain the concept of Executive Compensation.",
    "Please give me more detailed summary of the document?",
]
all_questions = [
    "Does the ignition button have multiple modes?",
    "Why does the other instance of my multi-instance qmgr seem to hang after a failover? Queue manager will not start after failover.",
    "Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.)?",
    "Explain the concept of blockchain.",
    "What is the capital of France?",
    "Do Surface Porosity and Pore Size Influence Mechanical Properties and Cellular Response to PEEK?",
    "How does a vaccine work?",
    "Tell me the step-by-step instruction for front-door installation.",
    "What are the risk factors for heart disease?",
    "What is the % change in total property and equipment from 2018 to 2019?",
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
        with gr.Column(scale=5):
            gr.Markdown("## Chat with document(s)")
            chat_history = gr.Textbox(label="Chat History", interactive=False, lines=20, elem_id="chat-history")
            with gr.Row():
                chat_input = gr.Textbox(label="Ask additional questions about the document...", show_label=False, placeholder="Ask additional questions about the document...", elem_id="chat-input", lines=3)
                chat_btn = gr.Button("🚀 Send", variant="primary", elem_id="send-button", scale=0)
            chat_btn.click(chat_response, inputs=[chat_input, doc_selector, chat_history], outputs=chat_history).then(
                lambda: "",  # Return an empty string to clear the chat_input
                outputs=chat_input
            )

        # Right Sidebar (Sample Questions & History)
        with gr.Column(scale=3):
            gr.Markdown("## Frequently asked questions:")
            with gr.Column():
                gr.Examples(
                    examples=example_questions,
                    inputs=chat_input,
                    label=""
                )
                question_dropdown = gr.Dropdown(
                    label="",
                    choices=all_questions,
                    interactive=True,
                    info="Choose a question from the dropdown to populate the query box."
                )

            gr.Markdown("## Logs")
            history = gr.Textbox(label="Previous Queries", interactive=False)

    gr.HTML("""
    <style>
    #chat-history textarea {
        max-height: 500px !important;
        overflow-y: auto !important;
    }
    </style>
    """)

if __name__ == "__main__":
    interface.launch()