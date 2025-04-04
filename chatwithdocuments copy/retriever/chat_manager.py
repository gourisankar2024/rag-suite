import logging
from typing import List
from globals import app_config

def chat_response(query: str, selected_docs: List[str], history: str) -> str:
    """
    Generate a chat response based on the user's query and selected documents.

    Args:
        query (str): The user's query.
        selected_docs (List[str]): List of selected document filenames from the dropdown.
        history (str): The chat history.
        model_name (str): The name of the LLM model to use for generation.

    Returns:
        str: Updated chat history with the new response.
    """
    if not query:
        return history + "\n" + "Response: Please enter a query." if history else "Response: Please enter a query."

    if not selected_docs:
        return history + "\n" + "LLM: Please select at least one document." if history else "Response: Please select at least one document."

    # Retrieve the top 5 chunks based on the query and selected documents
    top_k_results = app_config.doc_manager.retrieve_top_k(query, selected_docs, k=5)
    
    if not top_k_results:
        return history + "\n" + f"User: {query}\nLLM: No relevant information found in the selected documents." if history else f"User: {query}\nLLM: No relevant information found in the selected documents."

    # Send the top K results to the LLM to generate a response
    try:
        llm_response, source_docs = app_config.gen_llm.generate_response(query, top_k_results)
    except Exception as e:
        return history + "\n" + f"User: {query}\nLLM: Error generating response: {str(e)}" if history else f"User: {query}\nLLM: Error generating response: {str(e)}"

    # Format the response for the chat history
    response = f"{llm_response}\n"
    '''for i, doc in enumerate(source_docs, 1):
        doc_id = doc.metadata.get('doc_id', 'Unknown')
        filename = next((name for name, d_id in app_config.doc_manager.document_ids.items() if d_id == doc_id), 'Unknown')
        response += f"{i}. {filename}: {doc.page_content[:100]}...\n"'''

    return history + "\n" + f"User: {query}\nResponse: {response}" if history else f"User: {query}\nResponse: {response}"