import json
import logging

def compute_metrics(attributes, total_sentences):
    # Extract relevant information from attributes
    all_relevant_sentence_keys = attributes.get("all_relevant_sentence_keys", [])
    all_utilized_sentence_keys = attributes.get("all_utilized_sentence_keys", [])
    sentence_support_information = attributes.get("sentence_support_information", [])

    # Compute Context Relevance
    context_relevance = len(all_relevant_sentence_keys) / total_sentences if total_sentences else 0
    
    # Compute Context Utilization
    context_utilization = len(all_utilized_sentence_keys) / total_sentences if total_sentences else 0
    
    # Compute Completeness score
    Ri = set(all_relevant_sentence_keys)
    Ui = set(all_utilized_sentence_keys)

    completeness_score = len(Ri & Ui) / len(Ri) if len(Ri) else 0

    # Compute Adherence
    adherence = all(info.get("fully_supported", False) for info in sentence_support_information)
    #adherence = 1 if all(info.get("fully_supported", False) for info in sentence_support_information) else 0
    
    return {
        "Context Relevance": context_relevance,
        "Context Utilization": context_utilization,
        "Completeness Score": completeness_score,
        "Adherence": adherence
    }

def get_metrics(attributes, total_sentences):
    if attributes.content:
        try:
            result_content = attributes.content  # Access the content attribute
            # Extract the JSON part from the result_content
            json_start = result_content.find("{")
            json_end = result_content.rfind("}") + 1
            json_str = result_content[json_start:json_end]
            result_json = json.loads(json_str)
            # Compute metrics using the extracted attributes
            metrics = compute_metrics(result_json, total_sentences)
            logging.info(metrics)
            
            return metrics        
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e}")

def get_attributes_text(attributes):
        try:
            result_content = attributes.content  # Access the content attribute
            # Extract the JSON part from the result_content
            json_start = result_content.find("{")
            json_end = result_content.rfind("}") + 1
            json_str = result_content[json_start:json_end]
            result_json = json.loads(json_str)
            
            # Extract the required fields from json
            relevance_explanation = result_json.get("relevance_explanation", "N/A")
            all_relevant_sentence_keys = result_json.get("all_relevant_sentence_keys", [])
            overall_supported_explanation = result_json.get("overall_supported_explanation", "N/A")
            overall_supported = result_json.get("overall_supported", "N/A")
            sentence_support_information = result_json.get("sentence_support_information", [])
            all_utilized_sentence_keys = result_json.get("all_utilized_sentence_keys", [])

            # Format the metrics for display
            attributes_text = "Attributes:\n"
            attributes_text = f"### Relevance Explanation:\n{relevance_explanation}\n\n"
            attributes_text += f"### All Relevant Sentence Keys:\n{', '.join(all_relevant_sentence_keys)}\n\n"
            attributes_text += f"### Overall Supported Explanation:\n{overall_supported_explanation}\n\n"
            attributes_text += f"### Overall Supported:\n{overall_supported}\n\n"
            attributes_text += "### Sentence Support Information:\n"
            for info in sentence_support_information:
                attributes_text += f"- Response Sentence Key: {info.get('response_sentence_key', 'N/A')}\n"
                attributes_text += f"  Explanation: {info.get('explanation', 'N/A')}\n"
                attributes_text += f"  Supporting Sentence Keys: {', '.join(info.get('supporting_sentence_keys', []))}\n"
                attributes_text += f"  Fully Supported: {info.get('fully_supported', 'N/A')}\n"
            attributes_text += f"\n### All Utilized Sentence Keys:\n{', '.join(all_utilized_sentence_keys)}"

            return attributes_text
        except Exception as e:
            logging.error(f"Error extracting attributes: {e}")
            return f"An error occurred while extracting attributes: {e}"