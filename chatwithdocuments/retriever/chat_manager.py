from datetime import datetime
import logging
from typing import List
from globals import app_config

def chat_response(query: str, selected_docs: List[str], history: List[dict]) -> List[dict]:
    """
    Generate a chat response based on the user's query and selected documents.

    Args:
        query (str): The user's query.
        selected_docs (List[str]): List of selected document filenames from the dropdown.
        history (List[dict]): The chat history as a list of {'role': str, 'content': str} dictionaries.

    Returns:
        List[dict]: Updated chat history with the new response in 'messages' format.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Handle empty query
    if not query:
        return history + [{"role": "assistant", "content": "Please enter a query."}]

    # Handle no selected documents
    if not selected_docs:
        return history + [{"role": "assistant", "content": "Please select at least one document."}]

    # Retrieve the top 5 chunks based on the query and selected documents
    top_k_results = app_config.doc_manager.retrieve_top_k(query, selected_docs, k=5)
    
    if not top_k_results:
        return history + [
            {"role": "user", "content": f"{query}"},
            {"role": "assistant", "content": "No relevant information found in the selected documents."}
        ]

    # Send the top K results to the LLM to generate a response
    try:
        llm_response, source_docs = app_config.gen_llm.generate_response(query, top_k_results)
    except Exception as e:
        return history + [
            {"role": "user", "content": f"{query}"},
            {"role": "assistant", "content": f"Error generating response: {str(e)}"}
        ]

    # Format the response (uncomment and adapt if you want to include source docs)
    response = llm_response
    # for i, doc in enumerate(source_docs, 1):
    #     doc_id = doc.metadata.get('doc_id', 'Unknown')
    #     filename = next((name for name, d_id in app_config.doc_manager.document_ids.items() if d_id == doc_id), 'Unknown')
    #     response += f"\n{i}. {filename}: {doc.page_content[:100]}..."

    # Return updated history with new user query and LLM response
    return history + [
        {"role": "user", "content": f"{query}"},
        {"role": "assistant", "content": response}
    ]