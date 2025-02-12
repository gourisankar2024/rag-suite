import logging
import os
from langchain_groq import ChatGroq


def initialize_generation_llm(input_model_name):
    os.environ["GROQ_API_KEY"] = ""
    model_name = input_model_name    
    llm = ChatGroq(model=model_name, temperature=0.7)
    llm.name = model_name
    logging.info(f'Generation LLM {model_name} initialized')
    
    return llm


def initialize_validation_llm(input_model_name):
    os.environ["GROQ_API_KEY"] = ""
    model_name = input_model_name      
    llm = ChatGroq(model=model_name, temperature=0.7)
    llm.name = model_name
    logging.info(f'Validation LLM {model_name} initialized')
    
    return llm