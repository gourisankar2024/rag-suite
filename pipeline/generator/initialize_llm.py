import logging
import os
from langchain_groq import ChatGroq


def initialize_generation_llm(input_model_name):
    api_key = os.getenv("GROQ_API_KEY")  # Fetch from environment
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Please add it in Hugging Face Secrets.")
    
    os.environ["GROQ_API_KEY"] = api_key  # Explicitly set it
    model_name = input_model_name    
    llm = ChatGroq(model=model_name, temperature=0.7)
    llm.name = model_name
    logging.info(f'Generation LLM {model_name} initialized')
    
    return llm


def initialize_validation_llm(input_model_name):
    api_key = os.getenv("GROQ_API_KEY")  # Fetch from environment
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Please add it in Hugging Face Secrets.")
    
    os.environ["GROQ_API_KEY"] = api_key  # Explicitly set it
    model_name = input_model_name      
    llm = ChatGroq(model=model_name, temperature=0.7)
    llm.name = model_name
    logging.info(f'Validation LLM {model_name} initialized')
    
    return llm