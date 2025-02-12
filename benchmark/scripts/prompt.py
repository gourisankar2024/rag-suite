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