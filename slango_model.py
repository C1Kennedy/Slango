import pandas as pd
import numpy as np
import spacy

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

max_words = 10000

df = pd.read_csv('urbandict-word-defs.csv', error_bad_lines=False, nrows=max_words)

df.dropna()
df.drop_duplicates(subset=['word', 'definition'], inplace = True)

dataset = df[['word', 'up_votes', 'down_votes', 'definition']].copy()

print(dataset.head())

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([c for c in text if c not in ('!', '.', ',', '?', ':', ';', '"', "'", '-')])
    return text

def tokenize_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return tokens

#this may not be necessary?
dataset['definition'] = dataset['definition'].astype(str).apply(preprocess_text)
dataset['tokens'] = dataset['definition'].apply(tokenize_text)

definitions = dataset['tokens']
words = dataset['word']

# Tokenize words
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(words)
word_sequences = word_tokenizer.texts_to_sequences(words)

# Tokenize definitions
definition_tokenizer = Tokenizer()
definition_tokenizer.fit_on_texts(definitions)
definition_sequences = definition_tokenizer.texts_to_sequences(definitions)

# Define the model
max_word_sequence_length = max(len(seq) for seq in word_sequences)
max_definition_sequence_length = max(len(seq) for seq in definition_sequences)

model = Sequential()
model.add(Embedding(input_dim=len(definition_tokenizer.word_index) + 1, output_dim=128, input_length=max_definition_sequence_length))
model.add(LSTM(256))
model.add(Dense(len(word_tokenizer.word_index) + 1, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')