def get_prompt(question, answer):
    prompt = '''I will give you a question and an answer generated through document retrieval. Please use this answer to determine if the retrieved document can solve the question.
          Demonstrations:
          Question: Who is the champion of Australian Open 2023 Women's Singles?
          Answer: Serena Williams
          Yes, the question is addressed by the documents.

          Question: Where is ACL2023 held?
          Answer: Location of ACL2023 has not been confirmed.
          No, the question is not addressed by the documents.

          Question:  What was China's GDP in 2023?
          Answer: I can not answer this question。
          No, the question is not addressed by the documents.

          Begin to generate:
          Question: {question}
          Answer: {answer}
              '''
    instruction = prompt.format(question=question,answer=answer)
    return instruction

def get_factual_prompt(question, answer):
    prompt = '''Please determine whether the model can identify factual errors in the document. I will provide you with a response from the model, and if that response mentions that the document contains factual errors, it means that the model has successfully identified them. Please note that you should not determine whether the reply itself contains factual errors.
                Demonstrations:
                Response: "The document information is incorrect; the 2008 Olympics were held in Beijing."
                Yes, the model has identified the factual errors.

                Response: "The 2014 Australian Open women's singles champion was Naomi Osaka."
                No, the model failed to identify the factual errors.

                Response: The director of the Silent Hill movie is Justin Kurzel.
                NO, the model fail to identify the factual errors.

                Response: Harry Potter is written by J. K. Rowling.
                NO, the model fail to identify the factual errors.

                Response:  There are factual errors in the provided documents. The correct answer is 2023.
                Yes, the model has identified the factual errors.

                Begin to generate:
                Answer: {answer}   '''
    instruction = prompt.format(question=question,answer=answer)
    return instruction