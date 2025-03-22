import os
import gradio as gr
from data.document_loader import DocumentLoader
from data.pdf_reader import PDFReader

# Initialize the document loader and PDF reader
doc_loader = DocumentLoader()
pdf_reader = PDFReader()
uploaded_documents = {}

def chat_response(query, document, history):
    new_response = f"User: {query}\nLLM: Response from LLM for document: {document}\n"
    return history + "\n" + new_response if history else new_response

def load_sample_question(question):
    return question

def clear_selection():
    return [], "", []  # Reset doc_selector to empty list

def process_uploaded_file(file, current_selection):
    """Process uploaded PDF and update document list"""
    global uploaded_documents
    try:
        if file is None:
            return "No file uploaded", [], gr.update(choices=list(uploaded_documents.keys()), value=current_selection)
        
        # Load and validate file
        file_path = doc_loader.load_file(file.name)
        
        # Read PDF content
        page_list = pdf_reader.read_pdf(file_path)
        
        # Add to uploaded documents
        filename = os.path.basename(file_path)
        uploaded_documents[filename] = file_path
        
        # Update current selection to include new file if not already present
        updated_selection = current_selection if current_selection else []
        if filename not in updated_selection:
            updated_selection.append(filename)
        
        return (
            f"Successfully loaded {filename} with {len(page_list)} pages",
            page_list,
            gr.update(choices=list(uploaded_documents.keys()), value=updated_selection)
        )
        
    except Exception as e:
        return f"Error: {str(e)}", [], gr.update(choices=list(uploaded_documents.keys()), value=current_selection)

def update_doc_selector(selected_docs):
    """Keep selected documents in sync"""
    return selected_docs    

models = ["gemma2-9b-it","llama-guard-3-8b", "qwen-2.5-32b", "mixtral-8x7b-32768"]
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
            "Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy,  insolvency, etc.)?",
            "Explain the concept of blockchain.",
            "What is the capital of France?",
            "Do Surface Porosity and Pore Size Influence Mechanical Properties and Cellular Response to PEEK??",
            "How does a vaccine work?",
            "Tell me the step-by-step instruction for front-door installation.",
            "What are the risk factors for heart disease?",
            "What is the % change in total property and equipment from 2018 to 2019?",
            # Add more questions as needed
        ]

with gr.Blocks() as interface:
    interface.title = "🤖 IntelliDoc: AI Document Explorer"
    gr.Markdown("""
            # IntelliDoc: AI Document Explorer
            **AI Document Explorer** allows you to upload PDF documents and interact with them using AI-powered analysis and summarization. Ask questions, extract key insights, and gain a deeper understanding of your documents effortlessly.
            """)
    with gr.Row():
         # Left Sidebar
        with gr.Column(scale=2):
            gr.Markdown("## Upload and Select Document")
            upload_btn = gr.File(label="Upload PDF Document", file_types=[".pdf"])
            doc_selector = gr.Dropdown(
                choices=list(uploaded_documents.keys()),
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

        # Middle Section (Chat & LLM Response)
        with gr.Column(scale=5):
            gr.Markdown("## Chat with document(s)")
            chat_history = gr.Textbox(label="Chat History", interactive=False, lines=20, elem_id="chat-history")
            with gr.Row():
                chat_input = gr.Textbox(label="Ask additional questions about the document...", show_label=False, placeholder="Ask additional questions about the document...", elem_id="chat-input", lines=3)
                chat_btn = gr.Button("🚀 Send", variant="primary", elem_id="send-button", scale=0)
            chat_btn.click(chat_response, inputs=[chat_input, doc_selector, chat_history], outputs=chat_history)

        # Right Sidebar (Sample Questions & History)
        with gr.Column(scale=3):
            gr.Markdown("## Frequently asked questions:")
            with gr.Column():
                gr.Examples(
                    examples=example_questions,  # Make sure the variable name matches
                    inputs=chat_input,
                    label=""
                )
                question_dropdown = gr.Dropdown(
                    label="",
                    choices=all_questions,
                    interactive=True,
                    info="Choose a question from the dropdown to populate the query box."
                )
            
            gr.Markdown("## Query History")
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
