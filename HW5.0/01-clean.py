#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:03:10 2021

@author: ruitongliu
"""

import os
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk

dir = "/Users/ruitongliu/Desktop/590/Dataset"

data1 = ""
df = open(os.path.join(dir, "TheFighter.txt"))
data1 = df.read()
df.close()

data1_list = np.array(nltk.tokenize.sent_tokenize(data1))
data1_list = data1_list[:700]

data2 = ""
df = open(os.path.join(dir, "TheHouseofSpies.txt"))
data2 = df.read()
df.close()

data2_list = np.array(nltk.tokenize.sent_tokenize(data2))
data2_list = data2_list[:700]

data3 = ""
df = open(os.path.join(dir, "WhiteFang.txt"))
data3 = df.read()
df.close()

data3_list = np.array(nltk.tokenize.sent_tokenize(data3))
data3_list = data3_list[:700]

texts = []
labels = []

for i in range(700):
    texts.append(pioneer_list[i])
    labels.append(0)
    texts.append(ascanio_list[i])
    labels.append(1)
    texts.append(roadtobunker_list[i])
    labels.append(2)

maxlen = 100
training_samples = 1200
validation_samples = 300
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found %s unique tokens. " % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('shape of data tensor:', data.shape)
print('shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_test = data[training_samples: training_samples + validation_samples]
y_test = labels[training_samples: training_samples + validation_samples]

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1).astype('float32')
y_train = y_train.reshape(y_train.shape[0], 1, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1).astype('float32')
y_test = y_test.reshape(y_test.shape[0], 1, 1).astype('float32')

def load_data():
    return x_train, y_train, x_test, y_test
    
