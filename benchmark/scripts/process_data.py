import logging
import random
import math

def process_data(instance, noise_rate, passage_num, filename, correct_rate=0):
    print(filename)
    """Process the data for generating a noisy document set."""
    query = instance['query']
    ans = instance['answer']
    logging.info(f"Query: {query}")
    logging.info(f"Answer: {ans}")
    
    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num
    docs = []
    
    # Handling the '_int' case in filename
    if '_int' in filename:
        for i in instance['positive']:
            random.shuffle(i)
        docs = [i[0] for i in instance['positive']]
        if len(docs) < pos_num:
            maxnum = max([len(i) for i in instance['positive']])
            for i in range(1, maxnum):
                for j in instance['positive']:
                    if len(j) > i:
                        docs.append(j[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            negative = instance['negative'][:neg_num]
            docs += negative
    
    # Handling the '_fact' case in filename
    elif '_fact' in filename:
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(instance['positive'])))
        selected = random.sample(indexs, min(len(indexs), pos_num))
        docs = [instance['positive_wrong'][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        if correct_num > 0 and len(remain) > 0:
            docs += [instance['positive'][i] for i in random.sample(remain, min(len(remain), correct_num))]
        if neg_num > 0:
            docs += instance['negative'][:neg_num]
    
    # Default case (when filename doesn't match '_int' or '_fact')
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])

        positive = instance['positive'][:pos_num]
        negative = instance['negative'][:neg_num]

        docs = positive + negative
        # Count the positive and negative documents
        num_positive = sum(1 for doc in docs if doc in positive)
        num_negative = sum(1 for doc in docs if doc in negative)
        logging.info(f"Using {num_positive} positive and {num_negative} negative documents as context")
    
    # Shuffle the final document list
    random.shuffle(docs)
    return query, ans, docs
