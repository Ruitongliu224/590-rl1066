#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:19:32 2021

@author: ruitongliu
"""

import matplotlib.pyplot as plt
from keras import layers 
from keras import models
import numpy as np
import warnings

#data to use
data_sets = {"mnist":0,"mnist_fashion":1,"cifar10":2}

## Parameters:
data = 'mnist'
data_augmentation = True
epochs=20 
batch_size=64
model_type = 'CNN'
# model_type = 'ANN'

## determine which dataset to use
if data == 'mnist':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
if data == 'fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
if data == 'cifar10':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


#MODIFIED FROM CHOLLETT P120
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

if (dataset == 'mnist' or dataset == 'mnist_fashion'):
        train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
        test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
if (dataset == 'cifar10'):
        train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 3))
        test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 3))


#NORMALIZE
    NKEEP=10000
    batch_size=int(0.05*NKEEP)
    epochs=20
    print("batch_size",batch_size)
    train_images = train_images.astype('float32') / 255 
    test_images = test_images.astype('float32') / 255  
    
    #CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
    from tensorflow.keras.utils import to_categorical
    tmp=train_labels[0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print(tmp, '-->',train_labels[0])
    print("train_labels shape:", train_labels.shape)
    
    #SPLIT
    from sklearn.model_selection import train_test_split
    partial_train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=25)
    partial_train_labels, val_labels = train_test_split(train_labels, test_size=0.2, random_state=25)


#BUILD MODEL SEQUENTIALLY (LINEAR STACK) CNN
#-------------------------------------------

def CNN_model(height, width, channels):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)))
    model.add(layers.MaxPooling2D((2, 2)))
        
    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
        
    model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
    history = model.fit(train_images[train], train_labels[train], epochs = epochs, batch_size = batch_size)
        scores = model.evaluate(train_images[test], train_labels[test], batch_size = train_images[test].shape[0])
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_no += 1
    print("average accuracy = ", np.mean(acc_per_fold))
    print("average loss = ", np.mean(loss_per_fold))
    
def ANN_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
    history = model.fit(train_images[train], train_labels[train], epochs = epochs, batch_size = batch_size)
        scores = model.evaluate(train_images[test], train_labels[test], batch_size = train_images[test].shape[0])
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_no += 1
    print("average accuracy = ", np.mean(acc_per_fold))
    print("average loss = ", np.mean(loss_per_fold))


plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model training and validation accuracy history')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['Acuuracy', 'Loss'], loc='upper left')
plt.show()


#MODEL SAVING AND LOADING

from keras.models import load_model
model.save("my_model")

model = load_model("my_model")



#Visualize the CNN

if (dataset == 'mnist' or dataset =='mnist_fashion'):
        img = test_images[51].reshape(1,MNIST['height'],MNIST['width'],MNIST['channels'])
if (dataset == 'cifar10'):
        img = test_images[51].reshape(1,cifar_10['height'],cifar_10['width'],cifar_10['channels'])       
    
def visualiza_activations(): 

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_images[0])

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1],
                    scale * display_grid.shape[0]))
plt.title(layer_name)
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis') 
plt.show()











