import os
import random
import pandas as pd
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # supress tensorflow console logging

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import tensorflow as tf
from pickle import dump, load


TRAIN_LEN = 25
MODEL_PATH = "epochLSTM.h5"
TOKENIZER_PATH = "epochTK"
SEQUENCE_PATH = "epochSequece"


def gen_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    """
    Generates new text based on the seed text received

    Keyword arguments:
    model -- trained model used to predict words
    tokenizer -- trained tokenizer
    seq_len -- lenght of the seed text
    seed_text -- inital text to generate predictions from
    num_gen_words -- how many new words should be predicted
    """

    output_text = []
    input_text = seed_text

    for i in range(num_gen_words):
        # Take the input text string and encode it to a sequence
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating="pre")

        # Predict Class Probabilities for each word
        pred_word_ind = np.argmax(model.predict(pad_encoded), axis=-1)[0]

        pred_word = tokenizer.index_word[pred_word_ind]

        # Update the sequence of input text (shifting one over with the new word)
        input_text += " " + pred_word
        output_text.append(pred_word)
    return " ".join(output_text)


def main():
    # Load Objects
    model = load_model(MODEL_PATH)
    tokenizer = load(open(TOKENIZER_PATH, "rb"))
    text_sequences = load(open(SEQUENCE_PATH, "rb"))

    # Picks seed text randomly
    random.seed(os.urandom(3123131))
    random_pick = random.randint(0, len(text_sequences))

    random_seed_text = text_sequences[random_pick]
    seed_text = " ".join(random_seed_text)
    print("Seed Text: ", ''.join(seed_text))

    # Generates new text
    gen_txt = gen_text(
        model, tokenizer, TRAIN_LEN, seed_text=seed_text, num_gen_words=31
    )

    print("Generated Text: ", gen_txt)


if __name__ == "__main__":
    main()
