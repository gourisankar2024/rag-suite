from generator.create_prompt import create_prompt
from generator.document_utils import apply_sentence_keys_documents, apply_sentence_keys_response

# Function to extract attributes
def extract_attributes(val_llm, question, relevant_docs, response):
    # Format documents into a string by accessing the `page_content` attribute of each Document
    #formatted_documents = "\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(relevant_docs)])
    formatted_documents = apply_sentence_keys_documents(relevant_docs)
    formatted_responses = apply_sentence_keys_response(response)

    #print(f"Formatted documents : {formatted_documents}")
    # Print the number of sentences in each document
    '''for i, doc in enumerate(formatted_documents):
        num_sentences = len(doc)
        print(f"Document {i} has {num_sentences} sentences.")'''

    # Calculate the total number of sentences from formatted_documents
    total_sentences = sum(len(doc) for doc in formatted_documents)
    #print(f"Total number of sentences {total_sentences}")
    
    attribute_prompt = create_prompt(formatted_documents, question, formatted_responses)

    # Instead of using BaseMessage, pass the formatted prompt directly to invoke
    result = val_llm.invoke(attribute_prompt)

    return result, total_sentences