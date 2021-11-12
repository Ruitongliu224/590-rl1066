#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:30:08 2021

@author: ruitongliu
"""

from tensorflow import keras
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import auc
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import nltk

dir = '/Users/ruitongliu/Desktop/590'

clean_data = __import__('01-clean')
model_train = __import__('02-train')
x_train, y_train, x_test, y_test = clean_data.load_data()

maxlen = 100
max_words = 10000

# model type
CNN_model = model_train.build_CNN()
CNN_model.load_weights('1D_CNN_model.h5')
CNN_model.summary()

train_loss, train_acc = CNN_model.evaluate(x_train, y_train.reshape(y_train.shape[0]))
test_loss, test_acc = CNN_model.evaluate(x_test, y_test.reshape(y_test.shape[0]))
print('train_acc:', train_acc)
print('test_acc:', test_acc)

RDD_model = model_train.build_RNN()
RDD_model.load_weights('RNN_model.h5')
RDD_model.summary()
train_loss, train_acc = RDD_model.evaluate(x_train.reshape(len(x_train),-1), y_train.reshape(len(y_train),-1))
test_loss, test_acc = RDD_model.evaluate(x_test.reshape(len(x_test),-1), y_test.reshape(len(y_test),-1))
print('train_acc:', train_acc)
print('test_acc:', test_acc)