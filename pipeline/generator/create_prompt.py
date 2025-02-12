from langchain.docstore.document import Document

def create_prompt(documents, question, response):
    prompt = f""" I asked someone to answer a question based on one or more documents. Your task is to review their response and assess whether or not each sentence in that response is supported by text in the documents. If supported, identify which sentences in the documents provide that support. Additionally, identify which documents contain useful information for answering the question, and which documents the answer was sourced from.

    Here are the documents, each of which is split into sentences. Alongside each sentence is associated key, such as '0a.' or '0b.' that you can use to refer to it:
    {documents}

    The question was:
    {question}

    Here is their response, split into sentences with associated keys:
    {response}

    Provide a JSON response with an answer , relevance_explanation, all_relevant_sentence_keys, overall_supported_explanation, overall_supported, sentence_support_information(this can be a json inside a json and with these fields response_sentence_key, explanation, supporting_sentence_keys, fully_supported) and all_utilized_sentence_keys below are the definitions

    You must respond with a JSON object matching this schema:

    {{
      "answer": "string",
      "relevance_explanation": "string",
      "all_relevant_sentence_keys": ["string"],
      "overall_supported_explanation": "string",
      "overall_supported": "boolean",
      "sentence_support_information": [
        {{
          "response_sentence_key": "string",
          "explanation": "string",
          "supporting_sentence_keys": ["string"],
          "fully_supported": "boolean"
        }},
      ],
      "all_utilized_sentence_keys": ["string"]
    }}

        The relevance_explanation field is a string explaining which documents
        contain useful information for answering the question. Provide a step-by-step
        breakdown of information provided in the documents and how it is useful for
        answering the question.

        The all_relevant_sentence_keys field is a list of all document sentences keys
        (e.g. '0a') that are relevant to the question. Include every sentence that is
        useful and relevant to the question, even if it was not used in the response,
        or if only parts of the sentence are useful. Ignore the provided response when
        making this judgement and base your judgement solely on the provided documents
        and question. Omit sentences that, if removed from the document, would not
        impact someone's ability to answer the question.
        
        The overall_supported_explanation field is a string explaining why the response
        *as a whole* is or is not supported by the documents. In this field, provide a
        step-by-step breakdown of the claims made in the response and the support (or
        lack thereof) for those claims in the documents. Begin by assessing each claim
        separately, one by one; don’t make any remarks about the response as a whole
        until you have assessed all the claims in isolation.
        
        The overall_supported field is a boolean indicating whether the response as a
        whole is supported by the documents. This value should reflect the conclusion
        you drew at the end of your step-by-step breakdown in overall_supported_explanation.
        In the sentence_support_information field, provide information about the support
        *for each sentence* in the response.
        The sentence_support_information field is a list of objects, one for each sentence
        in the response. Each object MUST have the following fields:
        - response_sentence_key: a string identifying the sentence in the response.
        This key is the same as the one used in the response above.
        16
        - explanation: a string explaining why the sentence is or is not supported by the
        documents.
        - supporting_sentence_keys: keys (e.g. '0a') of sentences from the documents that
        support the response sentence. If the sentence is not supported, this list MUST
        be empty. If the sentence is supported, this list MUST contain one or more keys.
        In special cases where the sentence is supported, but not by any specific sentence,
        you can use the string "supported_without_sentence" to indicate that the sentence
        is generally supported by the documents. Consider cases where the sentence is
        expressing inability to answer the question due to lack of relevant information in
        the provided contex as "supported_without_sentence". In cases where the sentence
        is making a general statement (e.g. outlining the steps to produce an answer, or
        summarizing previously stated sentences, or a transition sentence), use the
        sting "general".In cases where the sentence is correctly stating a well-known fact,
        like a mathematical formula, use the string "well_known_fact". In cases where the
        sentence is performing numerical reasoning (e.g. addition, multiplication), use
        the string "numerical_reasoning".
        - fully_supported: a boolean indicating whether the sentence is fully supported by
        the documents.
        - This value should reflect the conclusion you drew at the end of your step-by-step
        breakdown in explanation.
        - If supporting_sentence_keys is an empty list, then fully_supported must be false.
        - Otherwise, use fully_supported to clarify whether everything in the response
        sentence is fully supported by the document text indicated in supporting_sentence_keys
        (fully_supported = true), or whether the sentence is only partially or incompletely
        supported by that document text (fully_supported = false).
        The all_utilized_sentence_keys field is a list of all sentences keys (e.g. ’0a’) that
        were used to construct the answer. Include every sentence that either directly supported
        the answer, or was implicitly used to construct the answer, even if it was not used
        in its entirety. Omit sentences that were not used, and could have been removed from
        the documents without affecting the answer.
        You must respond with a valid JSON string. Use escapes for quotes, e.g. ‘\\"‘, and
        newlines, e.g. ‘\\n‘. Do not write anything before or after the JSON string. Do not
        wrap the JSON string in backticks like ‘‘‘ or ‘‘‘json.
        As a reminder: your task is to review the response and assess which documents contain
        useful information pertaining to the question, and how each sentence in the response
        is supported by the text in the documents.

    """
    return prompt