import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv('fake_or_real_news.csv')
text = df.text.values[:5000]  # Limit to 5000 articles for memory efficiency

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")  # Limit vocabulary size
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

# Prepare input sequences
n_words = 10
input_sequences = []
next_words = []

for seq in sequences:
    for i in range(len(seq) - n_words):
        input_sequences.append(seq[i:i + n_words])
        next_words.append(seq[i + n_words])

# Convert to NumPy arrays
X = np.array(input_sequences)
y = np.array(next_words)

# Model definition
vocab_size = min(10000, len(tokenizer.word_index) + 1)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=n_words),
    LSTM(128, return_sequences=False),
    Dense(vocab_size, activation='softmax')
])

# Compile model
model.compile(optimizer=RMSprop(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, batch_size=128, epochs=5, shuffle=True)

# Save model
model.save("predict_next_word.h5")

# Reload model
model = tf.keras.models.load_model("predict_next_word.h5")


# Prediction function
def predict_next_word(input_text, n_best=5):
    input_seq = tokenizer.texts_to_sequences([input_text])[-1]
    input_seq = pad_sequences([input_seq], maxlen=n_words, padding="pre")

    predictions = model.predict(input_seq)[0]
    top_indices = np.argsort(predictions)[-n_best:]

    return [tokenizer.index_word[i] for i in top_indices if i in tokenizer.index_word]


# Text generation
def generate_text(input_text, text_length=100, creativity=3):
    word_sequence = input_text.split()
    for _ in range(text_length):
        sub_sequence = " ".join(word_sequence[-n_words:])
        try:
            choice = random.choice(predict_next_word(sub_sequence, creativity))
        except:
            choice = "<OOV>"  # Handle out-of-vocabulary words
        word_sequence.append(choice)

    return " ".join(word_sequence)


# Generate text
print(generate_text("trump", 50))
