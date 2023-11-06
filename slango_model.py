import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

dataset['definition'] = dataset['definition'].astype(str).apply(preprocess_text)
dataset['tokens'] = dataset['definition'].apply(tokenize_text)

definitions = dataset['tokens']
words = dataset['word']

class TextGenerationDataset(Dataset):
    def __init__(self, definitions, words, max_sequence_length):
        self.definitions = definitions
        self.words = words
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.definitions)
    
    def __getitem__(self, index):
        definitions_tokens = self.definitions.iloc[index]
        word_index = self.words.iloc[index]
        return{
            'defintion_tokens': definitions_tokens,
            'word_index': word_index
        }
    
text_generation_dataset = TextGenerationDataset(definitions, words, max_sequence_length = 20)

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextGenerationModel, self).__init()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
    
vocab_size = len(words)
embedding_dim = 128
hidden_dim = 256

model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

data_loader = DataLoader(text_generation_dataset, batch_size = 64, shuffle = True)

num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        definition_tokens = batch['definition_tokens']
        word_index = batch['word-index']

        optimizer.zero_grad()

        output = model(definition_tokens)
        output = output.view(-1, vocab_size)

        loss = criterion(output, word_index)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")

def generate_text(model, definition_tokens, max_length = 20):
    model.eval()
    generated_sequence = definition_tokens

    for _ in range(max_length):
        input_sequence = torch.LongTensor([definition_tokens])
        output = model(input_sequence)
        output = output.squeeze(0)
        predicted_word_index = torch.argmax(output).item()

        generated_sequence.append([predicted_word_index])
        definition_tokens.append(predicted_word_index)

    generated_text = " ".join([definition_tokens[idx] for idx in generated_sequence])

    return generated_text

