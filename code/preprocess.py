import random
import re
import numpy as np
import collections
import pandas as pd

def preprocess_captions(captions, window_size):
    for i, caption in enumerate(captions):
        # Taken from:
        # https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa

        # Convert the caption to lowercase, and then remove all special characters from it
        caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
      
        # Split the caption into separate words, and collect all words which are more than 
        # one character and which contain only alphabets (ie. discard words with mixed alpha-numerics)
        clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
      
        # Join those words into a string
        caption_new = ['<start>'] + clean_words[:window_size-1] + ['<end>']
      
        # Replace the old caption in the captions list with this new cleaned caption
        captions[i] = caption_new


def load_data(data_path):
    '''
    Method that was used to preprocess the data in the data.p file. You do not need 
    to use this method, nor is this used anywhere in the assignment. This is the method
    that the TAs used to pre-process the Flickr 8k dataset and create the data.p file 
    that is in your assignment folder. 
    Feel free to ignore this, but please read over this if you want a little more clairity 
    on how the images and captions were pre-processed 
    '''

    df = pd.read_csv(data_path)
    
    df = df.dropna()
    
    context = df.context.to_numpy()
    questions = df.question.to_numpy()
    answers = df.text.to_numpy()

    #randomly split examples into training and testing sets

    indices = np.arange(len(context))
    random.seed(0)
    np.random.shuffle(indices)
    train_context = context[indices][:int(len(context)*0.7)]
    train_questions = questions[indices][:int(len(questions)*0.7)]
    train_answers = answers[indices][:int(len(answers)*0.7)]
    test_context = context[indices][int(len(context)*0.7):]
    test_questions = questions[indices][int(len(questions)*0.7):]
    test_answers = answers[indices][int(len(answers)*0.7):]
    
    #remove special charachters and other nessesary preprocessing
    window_size = 20
    preprocess_captions(train_context, window_size)
    preprocess_captions(train_questions, window_size)
    preprocess_captions(train_answers, window_size)
    preprocess_captions(test_context, window_size)
    preprocess_captions(test_questions, window_size)
    preprocess_captions(test_answers, window_size)

    # count word frequencies and replace rare words with '<unk>'
    word_count = collections.Counter()
    for caption in train_context:
        word_count.update(caption)
    for caption in train_questions:
        word_count.update(caption)
    for caption in train_answers:
        word_count.update(caption)

    def unk_captions(captions, minimum_frequency):
        for caption in captions:
            for index, word in enumerate(caption):
                if word_count[word] <= minimum_frequency:
                    caption[index] = '<unk>'

    unk_captions(train_context, 50)
    unk_captions(train_questions, 50)
    unk_captions(train_answers, 50)
    
    unk_captions(test_context, 50)
    unk_captions(test_questions, 50)
    unk_captions(test_answers, 50)

    # pad captions so they all have equal length
    def pad_captions(captions, window_size):
        for caption in captions:
            caption += (window_size + 1 - len(caption)) * ['<pad>'] 
    
    pad_captions(train_context, window_size)
    pad_captions(train_questions, window_size)
    pad_captions(train_answers, window_size)
    
    pad_captions(test_context, window_size)
    pad_captions(test_questions, window_size)
    pad_captions(test_answers, window_size)

    # assign unique ids to every work left in the vocabulary
    word2idx = {}
    vocab_size = 0
    for caption in train_context:
        for index, word in enumerate(caption):
            if word in word2idx:
                caption[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                caption[index] = vocab_size
                vocab_size += 1
    for caption in train_questions:
        for index, word in enumerate(caption):
            if word in word2idx:
                caption[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                caption[index] = vocab_size
                vocab_size += 1
    for caption in train_answers:
        for index, word in enumerate(caption):
            if word in word2idx:
                caption[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                caption[index] = vocab_size
                vocab_size += 1
    for caption in test_context:
        for index, word in enumerate(caption):
            caption[index] = word2idx[word] 
    for caption in test_questions:
        for index, word in enumerate(caption):
            caption[index] = word2idx[word] 
    for caption in test_answers:
        for index, word in enumerate(caption):
            caption[index] = word2idx[word] 
            
    train_data = np.array([train_context, train_questions, train_answers])
    test_data = np.array([test_context, test_questions, test_answers])

    return dict(
        train_data          = np.array(train_data),
        test_data           = np.array(test_data),
        word2idx                = word2idx,
        idx2word                = {v:k for k,v in word2idx.items()},
    )