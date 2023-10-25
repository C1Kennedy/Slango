import pandas as pd
import numpy as np
import spacy

df = pd.read_csv('urbandict-word-defs.csv', error_bad_lines=False, nrows=10000)

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

dataset['definition'] = dataset['definition'].astype(str).apply(preprocess_text)
dataset['tokens'] = dataset['definition'].apply(tokenize_text)

print(dataset.head())