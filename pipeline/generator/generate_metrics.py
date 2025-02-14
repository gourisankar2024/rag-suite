import logging
import time
from generator.generate_response import generate_response
from retriever.retrieve_documents import retrieve_top_k_documents
from generator.compute_metrics import get_metrics
from generator.extract_attributes import extract_attributes

def retrieve_and_generate_response(gen_llm, vector_store, query):
    logging.info(f'Query: {query}')
    
    # Step 1: Retrieve relevant documents for given query
    relevant_docs = retrieve_top_k_documents(vector_store, query, top_k=5)
    #logging.info(f"Relevant documents retrieved :{len(relevant_docs)}")

    # Log each retrieved document individually
    #for i, doc in enumerate(relevant_docs):
        #logging.info(f"Relevant document {i+1}: {doc} \n")

    # Step 2: Generate a response using LLM
    response, source_docs = generate_response(gen_llm, vector_store, query, relevant_docs)

    logging.info(f"Response from LLM ({gen_llm.name}): {response}")

    return response, source_docs

def generate_metrics(val_llm, response, source_docs, query, time_to_wait):

    # Add a sleep interval to avoid hitting the rate limit
    time.sleep(time_to_wait)  # Adjust the sleep time as needed
    
    # Step 3: Extract attributes and total sentences for each query
    logging.info(f"Extracting attributes through validation LLM")
    attributes, total_sentences = extract_attributes(val_llm, query, source_docs, response)
    logging.info(f"Extracted attributes successfully")
    
    # Step 4 : Call the get metrics calculate metrics
    metrics = get_metrics(attributes, total_sentences)

    return attributes, metrics