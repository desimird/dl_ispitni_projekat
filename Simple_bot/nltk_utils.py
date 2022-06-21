import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sent, all_words):
    bag = np.zeros(len(all_words), dtype = np.float32)
    tokenized_sent = [stem(w) for w in tokenized_sent]
    
    for w in all_words:
        if w in tokenized_sent:
            bag[tokenized_sent.index(w)] = 1.0
            
    return bag
