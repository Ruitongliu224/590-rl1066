#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:04:32 2021

@author: ruitongliu
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.datasets import fashion_mnist

(train_X, train_Y), (test_X, test_y) = mnist.load_data()
(x_fashion, y_fashion), (fashion_test_X, fashion_test_y) = fashion_mnist.load_data()

train_X = train_X / np.max(train_X)
train_X = train_X.reshape(60000, 28 * 28)

test_X = test_X / np.max(test_X)
test_X = test_X.reshape(10000, 28 * 28)

x_fashion = x_fashion / np.max(x_fashion)
x_fashion = x_fashion.reshape(60000, 28 * 28)

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(28 * 28, activation='linear'))

model.compile(optimizer='adam',loss='mse')
model.summary()
fit = model.fit(train_X, train_X, epochs=10, batch_size=1000, validation_split=0.2)
model.save_weights('ae.h5')

plt.plot(fit.history["loss"], label="train_loss")
plt.plot(fit.history["val_loss"], label="val_loss")
plt.title("AE Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend(loc="lower left")
plt.savefig("result.png")

# test_loss = model.evaluate(x_fashion, x_fashion)
# print(test_loss)

pre_x = model.predict(test_X)
pre_x_fashion = model.predict(x_fashion)

test_X=test_X.reshape(len(test_X),28,28)
x_fashion=x_fashion.reshape(len(x_fashion),28,28)
pre_x = pre_x.reshape(len(test_X), 28, 28);
pre_x_fashion = pre_x_fashion.reshape(len(x_fashion), 28, 28);

f, ax = plt.subplots(4, 1)
index = 10
index2 = 100
ax[0].imshow(test_X[index])
ax[1].imshow(pre_x[index])
ax[2].imshow(x_fashion[index2])
ax[3].imshow(pre_x_fashion[index2])
plt.savefig("fashion.png")
