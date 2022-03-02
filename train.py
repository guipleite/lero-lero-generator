import numpy as np
import pandas as pd
import re
import os
import random
import spacy
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.utils import to_categorical
from pickle import dump, load
from tweet_scraper import get_tweets

trash = 'cunha depeduardocunha camaradeputados \n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n\r\n~" '  # Stripping tweets
l_ats = [
    "\\r\\n@depeduardocunha",
    "@depeduardocunha\\r\\n",
    "\\r\\n@camaradeputados",
    "cunha\n@depeduardocunha\n·\n1",
    'cunha\n@depeduardocunha\n·\n25',
    "@depdanicunha\ndani"
] 

START_DATE = "2015-01-01"
END_DATE = "2020-12-30"
ACCOUNT = "DepEduardoCunha"
TRAIN_LEN = 25
MODEL_PATH = "epochLSTM.h5"
TOKENIZER_PATH = "epochTK"
SEQUENCE_PATH = "epochSequece"

# Downloading trained pipelines for Portuguese if its not installed already
# !python -m spacy download pt_core_news_lg


def tokenize(df, trash, l_ats):
    """
    Returns a tokenizer and a tokenizer list of the received tweets.

    Keyword arguments:
    df -- pandas dataframe with tweets
    trash -- list with chars and symbols to remove from tweets
    l_ats -- list for removing common @s that cause a lot of bias in the final result
    """
    nlp = spacy.load("pt_core_news_lg", disable=["parser", "tagger", "ner"])

    doc_text = "".join(str(df["Embedded_text"].tolist()))
    # Removing unwanted punctuation and symbols
    tokens = [
        re.sub(r"^https?:\/\/.*[\r\n]*", "", token.text.lower(), flags=re.MULTILINE)
        for token in nlp(doc_text)
        if token.text not in trash
    ]
    tokens = list(filter(lambda a: a not in l_ats, tokens))
    print("Number of tokens found: ", len(tokens))
    print(tokens)

    # Organize into sequences of tokens
    text_sequences = [
        tokens[i - TRAIN_LEN + 1 : i] for i in range(TRAIN_LEN + 1, len(tokens))
    ]
    
    dump(text_sequences, open(SEQUENCE_PATH, "wb"))

    # Tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)
    sequences = np.array(tokenizer.texts_to_sequences(text_sequences))

    return tokenizer, sequences


def train_model(tokenizer, sequences):
    """
    Declares and trains a LSTM model

    Keyword arguments:
    tokenizer -- tokenizer used to indicate how many words have been found on tweets
    sequences -- tweet texts, used to train and validade the model
    """

    vocab_sz = len(tokenizer.word_counts)

    X = sequences[:, :-1]
    y = to_categorical(sequences[:, -1], num_classes=vocab_sz + 1)

    # Implementing a LSTM model
    model = Sequential()

    model.add(Embedding(vocab_sz + 1, TRAIN_LEN, input_length=TRAIN_LEN))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation="relu"))
    model.add(Dense(vocab_sz + 1, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.summary()

    model.fit(X, y, batch_size=128, epochs=290, verbose=1)

    return model


def run():

    df_tweets = get_tweets(START_DATE, END_DATE, account=ACCOUNT, words=None)
    # df_tweets = pd.read_csv('./data/tweets.csv')

    print("Tokenizing data ...")
    tokenizer, sequences = tokenize(df_tweets, trash, l_ats)
    
    print("Training model ...")
    model = train_model(tokenizer, sequences)

    # Saving the model and tokenizer
    model.save(MODEL_PATH)
    dump(tokenizer, open(TOKENIZER_PATH, "wb"))

    return model

if __name__ == "__main__":
    run()