# Cameron Kennedy, Blank Bruno, Bella Quintero, UTEP, Fall 2023
# Speech and Language Processing
# Project: Slango Model

import pandas as pd
import numpy as np
import random
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec

def preprocess_data(data):
    # Tokenize, lemmatize, stem, and return received data.
    tokenized_data = [
        word_tokenize(str(definition).lower()) + word_tokenize(str(word).lower())
        for word, definition in zip(data['word'], data['definition'])]
    
    lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    
    preprocessed_data = [
        [porter_stemmer.stem(lemmatizer.lemmatize(word)) for word in definition]
        for definition in tokenized_data]
    
    return preprocessed_data

def generate_text(word2vec_model, data, num_templates=1, num_words=20):
    # Randomly select a definition from the received data.
    entry = data.loc[random.randrange(0, len(data))]
    template_definition = entry['definition']

    # Tokenize the selected definition.
    tokenized_definition = word_tokenize(str(template_definition).lower())

    # Replace words in the sentence with similar words based on Word2Vec embeddings.
    for _ in range(num_words):
        placeholder = random.choice(tokenized_definition)
        replacement = replace_placeholder(word2vec_model, placeholder)
        tokenized_definition[tokenized_definition.index(placeholder)] = replacement

    # Return the word attached to the original definition, as well as the new definition.
    return entry['word'], ' '.join(tokenized_definition)

def replace_placeholder(word2vec_model, placeholder):
    # Try to replace the word with a word similar in meaning
    try:
        return word2vec_model.wv.most_similar(placeholder, topn=1)[0][0]
    
    except KeyError:
        # If unable, just return the word.
        return placeholder

def find_most_similar_words(word2vec_model, generated_sentence, training_data, top_n = 4):
    # Preprocess the generated sentence.
    preprocessed_generated_sentence = [word for word in word_tokenize(generated_sentence.lower()) if word in word2vec_model.wv.key_to_index]

    # Calculate the mean vector of the generated sentence.
    generated_sentence_vector = np.mean([get_word2vec_embeddings(word2vec_model, word) for word in preprocessed_generated_sentence], axis=0)

    # Initialize a list to store the top n most similar words and their cosine similarity scores.
    most_similar_words = []

    for index, definition in enumerate(training_data['definition']):
        tokenized_definition = [word for word in word_tokenize(str(definition).lower()) if word in word2vec_model.wv.key_to_index]
        definition_vector = np.mean([get_word2vec_embeddings(word2vec_model, word) for word in tokenized_definition], axis = 0)

        # Flatten the vectors before passing them to the cosine function.
        generated_sentence_vector = generated_sentence_vector.ravel()
        definition_vector = definition_vector.ravel()

        # Calculate the cosine similarity between the generated sentence vector and the current definition vector.
        cosine_similarity = 1 - cosine(generated_sentence_vector, definition_vector)

        # Add the current word (at the most similar definition's index) and its cosine similarity score to the most_similar_words list.
        most_similar_words.append((training_data['word'][index], cosine_similarity))

    # Shuffle the list of most similar words before returning them.
    random.shuffle(most_similar_words)

    # Return the top_n most similar shuffled words.
    return random.sample([(word, similarity) for word, similarity in most_similar_words[:top_n]], top_n)

def get_word2vec_embeddings(word2vec_model, word):
    try:
        return word2vec_model.wv[word]
    except KeyError:
        return np.zeros(word2vec_model.vector_size)
    
# Load data
train_data = pd.read_csv("urbandict-word-defs.csv", usecols=['word', 'definition'], nrows=10000)
gen_data = pd.read_csv("urbandict-word-defs.csv", usecols=['word', 'definition'], 
                       skiprows=range(1, 10000), nrows=10500-10000)

def main():
    #load Word2Vec model
    word2vec_model = Word2Vec.load("slango_word2vec.model")

    slang, generated_text = generate_text(word2vec_model, gen_data)
    print(f'Generated text: {generated_text}')

    similar_words = find_most_similar_words(word2vec_model, generated_text, train_data)
        
    index = random.randrange(0, len(similar_words))
    similar_words.insert(index, (slang, 0))
    
    for i, word in enumerate(similar_words):
        print(f"Word {i + 1}: {word[0]}")
        
    return index, slang.lower()

if __name__ == "__main__":
    main()