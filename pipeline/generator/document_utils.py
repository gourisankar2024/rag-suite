from typing import List

class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content

def apply_sentence_keys_documents(relevant_docs: List[Document]):
    result = []
    '''for i, doc in enumerate(relevant_docs):
        doc_id = str(i)
        title_passage = doc.page_content.split('\nPassage: ')
        title = title_passage[0]
        passages = title_passage[1].split('. ')
        
        doc_result = []
        doc_result.append([f"{doc_id}a", title])
        
        for j, passage in enumerate(passages):
            doc_result.append([f"{doc_id}{chr(98 + j)}", passage])
        
        result.append(doc_result)'''
    
    for relevant_doc_index, relevant_doc in enumerate(relevant_docs):
        sentences = []
        for sentence_index, sentence in enumerate(relevant_doc.page_content.split(".")):
            sentences.append([str(relevant_doc_index)+chr(97 + sentence_index), sentence])
        result.append(sentences)
    
    return result

def apply_sentence_keys_response(input_string):
    sentences = input_string.split('. ')
    result = [[chr(97 + i), sentence] for i, sentence in enumerate(sentences)]
    return result