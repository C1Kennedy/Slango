import pandas as pd
import numpy as np
from collections import Counter
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

max_words = 10000

df = pd.read_csv('urbandict-word-defs.csv', error_bad_lines=False, nrows=max_words)

df.dropna(inplace = True)
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

UNK_TOKEN = '<UNK>'
word_to_index = {word: index for index, word in enumerate(words)}
word_to_index[UNK_TOKEN] = len(words)  # Add UNK token to the vocabulary
index_to_word = {index: word for index, word in enumerate(words)}
index_to_word[len(words)] = UNK_TOKEN  # Update the index-to-word mapping

def tokens_to_indices(tokens):
    return [word_to_index.get(token, word_to_index[UNK_TOKEN]) for token in tokens]

# Apply the conversion to dataset
definitions_tokens_indices = definitions.apply(tokens_to_indices)

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

        word_index = word_to_index[word_index]
        return{
            'definition_tokens': definitions_tokens,
            'word_index': word_index
        }
    
text_generation_dataset = TextGenerationDataset(definitions_tokens_indices, words, max_sequence_length = 20)

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
    
vocab_size = len(index_to_word)
embedding_dim = 128
hidden_dim = 256

model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

def my_collate_fn(batch):
    definitions_tokens = [item['definition_tokens'] for item in batch]
    word_indices = [item['word_index'] for item in batch]
    
    # Pad the sequences to have the same length
    definitions_tokens_padded = pad_sequence([torch.LongTensor(seq) for seq in definitions_tokens], batch_first=True)
    word_indices = torch.LongTensor(word_indices)
    
    return {
        'definition_tokens': definitions_tokens_padded,
        'word_index': word_indices
    }


data_loader = DataLoader(text_generation_dataset, batch_size = 64, shuffle = True, collate_fn=my_collate_fn)

num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        definition_tokens = batch['definition_tokens']
        word_index = batch['word_index']
        word_index = word_index.view(-1, 1)

        optimizer.zero_grad()

        output = model(definition_tokens)
        output = output.view(definition_tokens.size(0), -1, vocab_size)

        # Flatten both the output and word_index tensors to calculate the loss
        loss = criterion(output.view(-1, vocab_size), word_index.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")

def generate_text(model, definition_tokens, max_length=20):
    model.eval()
    generated_sequence = definition_tokens.tolist()  # Convert to a Python list

    for _ in range(max_length):
        input_sequence = torch.LongTensor([generated_sequence[-1]])  # Use the generated sequence
        output = model(input_sequence)
        output = output.squeeze(0)
        predicted_word_index = torch.argmax(output).item()

        generated_sequence.append(predicted_word_index)

    generated_text = " ".join([index_to_word.get(idx, UNK_TOKEN) for idx in generated_sequence])

    return generated_text