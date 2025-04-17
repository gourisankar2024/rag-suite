from datetime import datetime
import logging
from typing import List


class ChatManager:
    def __init__(self, documentManager, llmManager):
        """
        Initialize the ChatManager.
        """
        self.doc_manager = documentManager
        self.llm_manager = llmManager

        logging.info("ChatManager initialized")

    def generate_chat_response(self, query: str, selected_docs: List[str], history: List[dict]) -> List[dict]:
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
        logging.info(f"Generating chat response for query: {query} at {timestamp}")

        # Handle empty query
        if not query:
            logging.warning("Empty query received")
            return history + [{"role": "assistant", "content": "Please enter a query."}]

        # Handle no selected documents
        if not selected_docs:
            logging.warning("No documents selected")
            return history + [{"role": "assistant", "content": "Please select at least one document."}]

        # Retrieve the top 5 chunks based on the query and selected documents
        try:
            top_k_results = self.doc_manager.retrieve_top_k(query, selected_docs, k=5)
        except Exception as e:
            logging.error(f"Error retrieving chunks: {str(e)}")
            return history + [
                {"role": "user", "content": f"{query}"},
                {"role": "assistant", "content": f"Error retrieving chunks: {str(e)}"}
            ]

        if not top_k_results:
            logging.info("No relevant chunks found")
            return history + [
                {"role": "user", "content": f"{query}"},
                {"role": "assistant", "content": "No relevant information found in the selected documents."}
            ]

        # Send the top K results to the LLM to generate a response
        try:
            llm_response, source_docs = self.llm_manager.generate_response(query, top_k_results)
        except Exception as e:
            logging.error(f"Error generating LLM response: {str(e)}")
            return history + [
                {"role": "user", "content": f"{query}"},
                {"role": "assistant", "content": f"Error generating response: {str(e)}"}
            ]

        # Format the response
        response = llm_response
        # Uncomment to include source docs in response (optional)
        # for i, doc in enumerate(source_docs, 1):
        #     doc_id = doc.metadata.get('doc_id', 'Unknown')
        #     filename = next((name for name, d_id in self.doc_manager.document_ids.items() if d_id == doc_id), 'Unknown')
        #     response += f"\n{i}. {filename}: {doc.page_content[:100]}..."

        logging.info("Chat response generated successfully")
        # Return updated history with new user query and LLM response
        return history + [
            {"role": "user", "content": f"{query}"},
            {"role": "assistant", "content": response}
        ]

    def generate_summary(self, chunks: any, summary_type: str = "medium") -> str:
        """
        Generate a summary of the selected documents.

        Args:
            selected_docs (List[str]): List of selected document filenames.
            summary_type (str): Type of summary ("small", "medium", "detailed").
            k (int): Number of chunks to retrieve from DocumentManager.
            include_toc (bool): Whether to include the table of contents (if available).

        Returns:
            str: Generated summary.

        Raises:
            ValueError: If summary_type is invalid or DocumentManager/LLM is not available.
        """
        if summary_type not in ["small", "medium", "detailed"]:
            raise ValueError("summary_type must be 'small', 'medium', or 'detailed'")

        if not chunks:
            logging.warning("No documents selected for summarization")
            return "Please select at least one document."

        
        llm_summary_response = self.llm_manager.generate_summary_v0(chunks = chunks)
        #logging.info(f" Summary response {llm_summary_response}")

        return llm_summary_response
    
    def generate_sample_questions(self, chunks: any):
        questions = self.llm_manager.generate_questions(chunks = chunks)
        return questions
