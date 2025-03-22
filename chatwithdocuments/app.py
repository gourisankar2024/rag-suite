import gradio as gr

def chat_response(query, document, history):
    new_response = f"User: {query}\nLLM: Response from LLM for document: {document}\n"
    return history + "\n" + new_response if history else new_response

def load_sample_question(question):
    return question

def clear_selection():
    return None

documents = [f"Document {i}" for i in range(1, 101)]  # Mock list of documents
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
            gr.Markdown("## Select Document")
            upload_btn = gr.File(label="Upload Document")
            doc_selector = gr.Dropdown(choices=documents, label="Documents", multiselect=True , info="Select one or more document(s) for analysis.")
            model_selector = gr.Dropdown(choices= models, label="Models", interactive=True, info="Select a model for response generation.")
            clear_btn = gr.Button("Clear Selection")
            clear_btn.click(clear_selection, outputs=doc_selector)

        # Middle Section (Chat & LLM Response)
        with gr.Column(scale=5):
            gr.Markdown("## Chat with document(s)")
            chat_history = gr.Textbox(label="Chat History", interactive=False, lines=20, elem_id="chat-history")
            with gr.Row():
                chat_input = gr.Textbox(label="Ask additional questions about the document...", show_label=False, placeholder="Ask additional questions about the document...", elem_id="chat-input", lines= 3)
                chat_btn = gr.Button("🚀 Send", variant="primary", elem_id="send-button", scale= 0)
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
