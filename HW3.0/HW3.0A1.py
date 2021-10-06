#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:51:49 2021

@author: ruitongliu
"""

from keras.datasets import boston_housing

(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std
test_data -= mean
test_data /= std

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation= 'relu',
                           input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64,activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer = "rmsprop",loss = "mse", metrics = ['mae'])
    return model

import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 1000
all_mae_histories = []
all_train_mae=[]
all_loss_values = []
all_val_values = []

for i in range(k):
    print('processing fold #', i)
    
    # Prepares the validation data: data from partition #k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] 
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    # Prepares the training data: data from all other partitions
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], 
                                         axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]], 
                                            axis=0)

    model = build_model()
    
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0) 
    train_mae_history = history.history['mae']
    mae_history = history.history['val_mae']
    loss_history = history.history['loss']
    val_history = history.history['val_loss']
    all_train_mae.append(train_mae_history)
    all_mae_histories.append(mae_history)
    all_loss_values.append(loss_history)
    all_val_values.append(val_history)
    
# Building the history of successive mean K-fold validation scores
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

average_loss_history = [
    np.mean([x[i] for x in all_loss_values]) for i in range(num_epochs)]

average_val_history = [
    np.mean([x[i] for x in all_val_values]) for i in range(num_epochs)]


#import matplotlib.pyplot as plt
def smooth_curve(points, factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])
smooth_loss_history = smooth_curve(average_loss_history[10:])
smooth_val_history = smooth_curve(average_val_history[10:])

#plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
#plt.xlabel('Epochs')
#plt.ylabel('Validation MAE')
#plt.show()

model = build_model()
model.fit(train_data,train_targets,
          epochs = 80,batch_size = 16,verbose = 0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)


