#importing libs

import json
from nltk_utils import tokenize, stem
import numpy as np
import random
import tflearn
import tensorflow as tf
import pickle

#importing libs

ERROR_THRESHOLD = 0.25

#functions needed for bot to work


#preprocessing input from user(functions needed for that)

def clean_up_sentence(sentence):
    sentence_words = tokenize(sentence)
    sentence_words = [stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):  # dobija se vektor 1 i 0 isto kao kada se pravio trening set
    
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1.0
                

    return(np.array(bag))

#preprocessing input from user(functions needed for that)

#answering the questions

def classify(sentence):
    results = model.predict([bow(sentence, all_words)])[0]

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
  
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((tags[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence):
    results = classify(sentence)
    
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    return print(random.choice(i['responses']))

            results.pop(0)


#answering the questions

#loading model and its data

# restore all of data structures

data = pickle.load( open( "training_data", "rb" ) )
all_words = data['words']
tags = data['classes']
X_train = data['train_x']
Y_train = data['train_y']

# import intents file

with open('intents.json', 'r') as f:
    intents = json.load(f)


#restoring model

NN = tflearn.input_data(shape=[None, len(X_train[0])])
NN = tflearn.fully_connected(NN, 8)
NN = tflearn.fully_connected(NN, 8)
NN = tflearn.fully_connected(NN, len(Y_train[0]), activation='softmax')
NN = tflearn.regression(NN)


model = tflearn.DNN(NN, tensorboard_dir='tflearn_logs')
model.load('./model.tflearn')

#chating with bot named Milorad :D

bot_name = "Milorad"
print("Let's chat!")
while True:
    sentence = input("You: ")
    print(bot_name, ":")
    response(sentence)
      
    a = classify(sentence)
    if a[0][0] == 'goodbye':
        break
