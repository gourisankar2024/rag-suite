import logging
import random
import math

def process_data(instance, noise_rate, passage_num, filename, correct_rate=0):
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
        # Adjust correct_num to not exceed passage_num - neg_num, excluding positive_wrong
        if correct_rate == 1.0:
            # For factual-only with noise, use only positive and negative documents
            correct_num = min(correct_num, passage_num - neg_num)
            pos_num = 0  # No positive_wrong documents when correct_rate = 1.0
        else:
            # For other correct_rate values, calculate pos_num for positive_wrong
            pos_num = passage_num - neg_num - correct_num
            if pos_num < 0:
                pos_num = 0  # Ensure pos_num is not negative

        # Select positive documents (factual) first
        indexs_positive = list(range(len(instance['positive'])))
        selected_positive = random.sample(indexs_positive, min(len(indexs_positive), correct_num))
        docs = [instance['positive'][i] for i in selected_positive]

        # Add negative documents (noise) if needed
        if neg_num > 0 and 'negative' in instance:
            docs += instance['negative'][:min(neg_num, len(instance['negative']))]

        # Only add positive_wrong documents if pos_num > 0 and correct_rate < 1.0
        if pos_num > 0 and correct_rate < 1.0:
            indexs_positive_wrong = list(range(len(instance['positive_wrong'])))
            selected_positive_wrong = random.sample(indexs_positive_wrong, min(len(indexs_positive_wrong), pos_num))
            docs += [instance['positive_wrong'][i] for i in selected_positive_wrong]

        # Ensure docs length does not exceed passage_num
        if len(docs) > passage_num:
            random.shuffle(docs)
            docs = docs[:passage_num]
        elif len(docs) < passage_num and 'negative' in instance:
            remaining = passage_num - len(docs)
            docs += instance['negative'][:min(remaining, len(instance['negative']))]
    
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
