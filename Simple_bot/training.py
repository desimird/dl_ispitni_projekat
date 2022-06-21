# importing libs

import json
from nltk_utils import tokenize, stem
import numpy as np
import random
import tflearn
import tensorflow as tf
import pickle

# importing libs

#getting training data

#loading json 
with open('intents.json', 'r') as f:
    intents = json.load(f)
    
#preprocessing data

all_words = []
tags = []
xy = []

ignore_words = ['?'] 

X_train = []
Y_train = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)  # extend zato sto je w niz pa da ne bi alL_words bio niz nizova vec samo 1d niz
        xy.append((w, tag))

        
all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(list(set(all_words)))
tags = sorted(list(set(tags)))


# creating training data
training = []
output = []
output_empty = [0] * len(tags)


for doc in xy:

    bag = []
    pattern_words = doc[0]
    pattern_words = [stem(word.lower()) for word in pattern_words]
    
    for w in all_words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[tags.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle features and turn into np.array
random.shuffle(training)
training = np.array(training,dtype=object)

# create train and test lists
X_train = list(training[:,0])
Y_train = list(training[:,1])


#getting training data

# making model and network

NN = tflearn.input_data(shape=[None, len(X_train[0])])
NN = tflearn.fully_connected(NN, 8)
NN = tflearn.fully_connected(NN, 8)
NN = tflearn.fully_connected(NN, len(Y_train[0]), activation='softmax')
NN = tflearn.regression(NN)

model = tflearn.DNN(NN, tensorboard_dir='tflearn_logs')

#training
model.fit(X_train, Y_train, n_epoch=1000, batch_size=1, show_metric=True)
model.save('model.tflearn')


#saving data and model parameters
pickle.dump( {'words':all_words, 'classes':tags, 'train_x':X_train, 'train_y':Y_train}, open( "training_data", "wb" ) )
