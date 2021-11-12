#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:03:10 2021

@author: ruitongliu
"""

import os
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Flatten, Dense
from keras import layers
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense,SimpleRNN, LSTM
from tensorflow.keras import regularizers


clean_data = __import__('01-clean')
x_train, y_train, x_test, y_test = clean_data.load_data()


embedding_dim = 100


def build_CNN():
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


# train RNN model
def build_RNN():
    model = models.Sequential()
    model.add(layers.Embedding(10000, 1))
    model.add(layers.SimpleRNN(1, return_sequences=True))
    model.add(layers.SimpleRNN(1, return_sequences=True))
    model.add(layers.SimpleRNN(1, return_sequences=True))
    model.add(layers.SimpleRNN(1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model


model_type = ['1D_CNN', 'RNN']

for model_type in model_type:
    if model_type == '1D_CNN':
        model = build_CNN()
        print(x_train.shape)
        print(y_train.shape)
        history = model.fit(x_train, y_train.reshape(y_train.shape[0]),
                            epochs=10,
                            batch_size=32,
                            validation_split=0.5)

        model.summary()

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('1D CNN: Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('1D CNN: Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        model.save_weights('1D_CNN_model.h5')


    if model_type == 'RNN':
        model = build_RNN()
        history = model.fit(x_train.reshape(len(x_train),-1), y_train.reshape(len(y_train),-1),
                            epochs=10,
                            batch_size=32,
                            validation_split=0.5)

        model.summary()

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('RNN: Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('RNN: Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        model.save_weights('RNN_model.h5')
