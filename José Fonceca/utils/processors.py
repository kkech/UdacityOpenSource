from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from random import sample

import numpy as np
import os
import pandas as pd
import pickle
import re
import torch as th
import torch.nn as tnn

from .constants import CLEANING_REGEX, NEGATIONS_DICT


def extend_data(X, Y, context_size):
    new_X = []
    new_Y = []
    for idx, x in enumerate(X):
        y = Y[idx]        
        for idx2, word in enumerate(x):
            if idx2 < 5:                
                new_X.append(pad_sequences([x[:idx2 + 1]], maxlen = context_size)[0])
            else:                
                new_X.append(x[idx2 - context_size + 1: idx2 + 1])
            new_Y.append(np.array(y[idx2], dtype = "int32"))
    new_X = np.vstack(new_X)
    new_Y = np.array(new_Y).astype(np.int32)
    return new_X, new_Y


def get_cleaned_text(text, stop_words, stemmer, stem = False):    
    neg_pattern = re.compile(r'\b(' + '|'.join(NEGATIONS_DICT.keys()) + r')\b')    
    text = re.sub(CLEANING_REGEX, " ", str(text).lower()).strip()
    text = neg_pattern.sub(lambda x: NEGATIONS_DICT[x.group()], text)
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
                continue
            tokens.append(token)    
    text = " ".join(tokens)
    text = re.sub("n't", "not", text)
    return re.sub("'s", "is", text)


def get_local_and_remote_data(data, local_share = 0.1):
    unique_users = data.user.unique()
    idx_share = int(local_share*unique_users.shape[0])
    local_users = unique_users[:idx_share]
    remote_users = unique_users[idx_share:]
    local_data = data[data.user.isin(local_users)]
    remote_data = data[data.user.isin(remote_users)]
    return local_data, remote_data


def index_data_by_date(data, string_tz = "PDT"):
    timezone = 'US/Pacific' if "PDT" or "PT" in string_tz else "UTC"
    data.date = data.date.str.replace(string_tz, "")
    data.date = data.date.astype("datetime64[ns]")
    data.index = data.date
    data.drop(["date"], axis = 1, inplace = True)
    data.index = data.index.tz_localize(timezone)
    return data


def merge_and_index_data(input_file, dump_file, word2idx_file, min_tweets = 20):        
    if os.path.isfile(dump_file) and os.path.isfile(word2idx_file):
        data = pd.read_pickle(dump_file)
        with open(word2idx_file, "rb") as f:
            word2idx = pickle.load(f)
        return data, word2idx

    columns = ["target", "ids", "date", "flag", "user", "text"]
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    data = pd.read_csv(input_file, encoding = "ISO-8859-1", header = None, 
                            names = columns)
    data.drop(["target", "flag", "ids"], axis = 1, inplace = True)
    users = data.groupby(by = "user").apply(len) > min_tweets
    data = data[data.user.isin(users[users].index)]
    data = index_data_by_date(data)
    data["cleaned_text"] = data.text.apply(lambda x: get_cleaned_text(x, stop_words, stemmer))
    data.drop_duplicates(subset = ["cleaned_text"], keep = False, inplace = True)
    sequences, tokenizer = text_to_sequence(data.cleaned_text, Tokenizer)
    data["sequence"] = sequences
    data = data[data.sequence.map(lambda x: len(x)) > 0]    
    data = data.merge(data.sequence.apply(lambda x: split_X_and_Y(x)), 
                    left_index = True, right_index = True)
    data.to_pickle(dump_file)    
    with open(word2idx_file, "wb") as f:
        pickle.dump(tokenizer.word_index, f, pickle.HIGHEST_PROTOCOL)
    return data, tokenizer.word_index


def split_X_and_Y(sequence):
    X = [0]    
    Y = [sequence[0]]
    for idx, token in enumerate(sequence[:-1]):
        X.append(token)
        Y.append(sequence[idx + 1])
    return pd.Series({"X": X, "Y": Y})


def text_to_sequence(texts, tokenizer):
    tokenizer = tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer.texts_to_sequences(texts), tokenizer


def print_predictions(sentence, model):
    sequence = [model.word2idx[word] for word in sentence]
    splitted = split_X_and_Y(sequence)    
    X, Y = extend_data([splitted.X], [splitted.Y], model.context_size)
    X, Y = th.tensor(X), th.tensor(Y)
    model.h_lstm = model._init_hidden(len(X))
    preds = tnn.functional.softmax(model.forward(X), dim = 1).topk(3, dim = 1)[1]
    for idx, pred in enumerate(preds[1:-1]):
        word_preds = []        
        for idx2 in pred:
            word_preds.append(model.idx2word[int(idx2)])            
        prev_word = sentence[idx]
        expected_word = sentence[idx + 1]
        print("Previous word: {} \t Expected word: {} \t Predictions: {}\t{}\t{}".format(prev_word, expected_word, *word_preds))
    