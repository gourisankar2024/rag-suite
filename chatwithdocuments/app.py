import logging
import gradio as gr
from utils.document_utils import initialize_logging
from globals import app_config 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
initialize_logging()

def load_sample_question(question):
    return question

def clear_selection():
    return gr.update(value=[]), "", ""  # Reset doc_selector to empty list

def process_uploaded_file(file, current_selection):
    """Process uploaded file using DocumentManager and update UI."""
    try:
        if file is None:
            # When file input is cleared, preserve current selection and choices
            uploaded_docs = app_config.doc_manager.get_uploaded_documents()
            return (
                "",
                gr.update(choices=uploaded_docs, value=current_selection or []),
                False,
                ""
            )
        
        status, filename, doc_id = app_config.doc_manager.process_document(file.name if file else None)

        updated_selection = current_selection if current_selection else []
        if filename and filename not in updated_selection:
            updated_selection.append(filename)
        trigger_summary = bool(filename)
        logging.info(f"Processed file: {filename}, Trigger summary: {trigger_summary}")

        return (
            status,
            gr.update(choices=app_config.doc_manager.get_uploaded_documents(), value=updated_selection),
            trigger_summary,
            filename  
        )
    except Exception as e:
        logging.error(f"Error in process_uploaded_file: {e}")
        return "Error processing file", gr.update(choices=[]), False, ''

def update_doc_selector(selected_docs):
    """Keep selected documents in sync."""
    return selected_docs

# UI Configuration
models = [ "gemma2-9b-it", "llama3-70b-8192"]

example_questions = [
    "What is the architecture of the Communication Server?",
    "Show me an example of a configuration file.",
    "How to create Protected File Directories ?",
    "What functionalities are available in the Communication Server setups?",
    "What is Mediator help?",
    "Why AzureBlobStorage port is used?"
]

with gr.Blocks(css="""
        .chatbot .user {
            position: relative;
            background-color: #cfdcfd;
            padding: 12px 16px;
            border-radius: 20px;
            border-bottom-right-radius: 6px;
            display: inline-block;
            max-width: 80%;
            margin: 8px 0;
        }

        /* Tail effect */
        .chatbot .user::after {
            content: "";
            position: absolute;
            right: -10px;
            bottom: 10px;
            width: 0;
            height: 0;
            border: 10px solid transparent;
            border-left-color: #cfdcfd;
            border-right: 0;
            border-top: 0;
            margin-top: -5px;
        }
        .chatbot .bot { background-color: #f1f8e9; padding: 8px; border-radius: 10px; }   /* Light green for bot responses */
    """) as interface:
    interface.title = "🤖 IntelliDoc: AI Document Explorer"
    gr.Markdown("""
        # 🤖 IntelliDoc: AI Document Explorer
        **AI Document Explorer** allows you to upload PDF documents and interact with them using AI-powered analysis and summarization. Ask questions, extract key insights, and gain a deeper understanding of your documents effortlessly.
    """)
    summary_query_state = gr.State()  # State to hold the summary query
    trigger_summary_state = gr.State()  # State to hold trigger flag
    filename_state = gr.State()  # State to hold file name
    chunks_state = gr.State()
    summary_text_state = gr.State()
    sample_questions_state = gr.State()

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
            upload_event = upload_btn.change(
                process_uploaded_file,
                inputs=[upload_btn, doc_selector],
                outputs=[
                    upload_status,
                    doc_selector,
                    trigger_summary_state,  # Store trigger_summary
                    filename_state
                ]
            )
            clear_btn.click(
                clear_selection,
                outputs=[doc_selector, upload_status, filename_state]
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
            chat_history = gr.Chatbot(label="Chat History", height= 650, bubble_full_width= False, type="messages")
            with gr.Row():
                chat_input = gr.Textbox(label="Ask additional questions about the document...", show_label=False, placeholder="Ask additional questions about the document...", elem_id="chat-input", lines=3)
                chat_btn = gr.Button("🚀 Send", variant="primary", elem_id="send-button", scale=0)
            chat_btn.click(app_config.chat_manager.generate_chat_response, inputs=[chat_input, doc_selector, chat_history], outputs=chat_history).then(
                lambda: "",  # Return an empty string to clear the chat_input
                outputs=chat_input
            )

        # Right Sidebar (Sample Questions & History)
        with gr.Column(scale=2):
            gr.Markdown("## Sample questions for this document:")
            with gr.Column():
                sample_questions = gr.Dropdown(
                    label="Select a sample question",
                    choices=[],
                    interactive=True,
                    allow_custom_value=True  # Allows users to type custom questions if needed
                )
                '''question_dropdown = gr.Dropdown(
                    label="",
                    choices=all_questions,
                    interactive=True,
                    info="Choose a question from the dropdown to populate the query box."
                )'''

           # After upload, generate "Auto Summary" message only if trigger_summary is True
            upload_event.then(
                fn=lambda trigger, filename: "Can you provide summary of the document" if trigger and filename else None,
                inputs=[trigger_summary_state, filename_state],
                outputs=[summary_query_state]
                ).then(
                    fn=lambda query, history: history + [{"role": "user", "content": ""}, {"role": "assistant", "content": "Generating summary of the document, please wait..."}] if query else history,
                    inputs=[summary_query_state, chat_history],
                    outputs=[chat_history]
                ).then(
                    fn=lambda trigger, filename: app_config.doc_manager.get_chunks(filename) if trigger and filename else None,
                    inputs=[trigger_summary_state, filename_state],
                    outputs=[chunks_state]
                ).then(
                    fn=lambda chunks: app_config.chat_manager.generate_summary(chunks) if chunks else None,
                    inputs=[chunks_state],
                    outputs=[summary_text_state]
                ).then(
                    fn=lambda summary, history: history + [{"role": "assistant", "content": summary}] if summary else history,
                    inputs=[summary_text_state, chat_history],
                    outputs=[chat_history]
                ).then(
                    fn=lambda chunks: app_config.chat_manager.generate_sample_questions(chunks) if chunks else [],
                    inputs=[chunks_state],
                    outputs=[sample_questions_state]
                ).then(
                    fn=lambda questions: gr.update(choices=questions if questions else ["No questions available"]),
                    inputs=[sample_questions_state],
                    outputs=[sample_questions]
                )
            # Populate chat_input when a question is selected
            sample_questions.change(
                fn=lambda question: question,
                inputs=[sample_questions],
                outputs=[chat_input]
            )
            #gr.Markdown("## Logs")
            #history = gr.Textbox(label="Previous Queries", interactive=False)

if __name__ == "__main__":
    interface.launch()