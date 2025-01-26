import random

import numpy as np
import pandas as pd
from keras.api.layers import LSTM, Dense, Input
from keras.api.models import Sequential, load_model
from keras.api.optimizers import RMSprop
from nltk.tokenize import RegexpTokenizer

df = pd.read_csv('fake_or_real_news.csv')
text = list(df.text.values)

joined_text = " ".join(text)
partial_text = joined_text[:1000000]

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(partial_text.lower())

unique_tokens = np.unique(tokens, return_counts=True)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens[0])}

n_words = 10
input_words = []
next_words = []
for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])

X = np.zeros((len(input_words), n_words, len(unique_tokens[0])), dtype=bool)
y = np.zeros((len(next_words), len(unique_tokens[0])), dtype=bool)

for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        X[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_words[i]]] = 1

model = Sequential([
    Input(shape=(n_words, len(unique_tokens[0]))),
    LSTM(128, return_sequences=True),
    Dense(len(unique_tokens[0]), activation='softmax')
])

model.compile(optimizer=RMSprop(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, batch_size=128, epochs=10, shuffle=True)

model.save("predict_next_word.h5")

model = load_model("predict_next_word.h5")


def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    x = np.zeros((1, n_words, len(unique_tokens[0])))
    for i, word in enumerate(input_text.split()[-n_words:]):
        x[0, i, unique_token_index[word]] = 1
    predictions = model.predict(x)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]


possible = predict_next_word("trump", 5)


def generate_text(input_text, text_length, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current + n_words])
        try:
            choice = unique_tokens[0][random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens[0])
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)


generate_text("trump", 100)
