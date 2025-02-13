
# Check if the predicted answer matches the ground truth
def check_answer(prediction, ground_truth):
    prediction = prediction.lower()
    if type(ground_truth) is not list:
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if isinstance(instance, list):
            flag = False
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels

# Evaluate if the result is correct (non-zero indicates correctness)
def get_evaluation(results):
    return 0 not in results

# Generate prediction based on query, documents, and model
def predict(query, ground_truth, docs, model, instruction, temperature):
    '''
    label: 0 for positive, 1 for negative, -1 for not enough information
    '''
    system_message = (
        'You are an accurate and reliable AI assistant that can answer questions with the help of external documents. '
        'Please note that external documents may contain noisy or factually incorrect information. If the information '
        'in the document contains the correct answer, you will give an accurate answer. If the information in the '
        'document does not contain the answer, you will generate "I can not answer the question because of the insufficient information in documents." '
        'If there are inconsistencies with the facts in some of the documents, please generate the response: "There are factual errors in the provided documents and provide the correct answer."'
    )

    if len(docs) == 0:
        text = instruction.format(QUERY=query, DOCS='')
        prediction = model.generate(text, temperature)
    else:
        docs = '\n'.join(docs)
        text = instruction.format(QUERY=query, DOCS=docs)
        prediction = model.generate(text, temperature, system_message)

    # Check if the prediction contains the 'insufficient information' phrase
    if 'insufficient information' in prediction:
        labels = [-1]
    else:
        labels = check_answer(prediction, ground_truth)

    # Check for factual errors in the prediction
    fact_label = 0
    if 'factual errors' in prediction:
        fact_label = 1

    return labels, prediction, fact_label
