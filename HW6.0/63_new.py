#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:17:44 2021

@author: ruitongliu
"""


from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.layers import Input,Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.datasets import cifar10
from keras.datasets import cifar100

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

classes=10

(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()
(x_fashion, y_fashion), (x_fashion_test, y_fashion_test) = cifar100.load_data()


train_X = train_X.astype('float32') / 255.
train_X = np.reshape(train_X, (-1, 32, 32, 3))


test_X = test_X.astype('float32') / 255.
test_X = np.reshape(test_X, (-1, 32, 32, 3))

x_fashion = x_fashion.astype('float32') / 255.
x_fashion = np.reshape(x_fashion, (-1, 32, 32, 3))

input_img = Input(shape=(32, 32, 3))
x = Conv2D(32,(3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)


x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

fit=autoencoder.fit(train_X, train_X, epochs=20
                    , batch_size=1000, shuffle=True,
                        validation_data=(test_X, test_X))

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

print(autoencoder.summary())
plt.plot(fit.history["loss"], label="train_loss")
plt.plot(fit.history["val_loss"], label="val_loss")
plt.title("CNN Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend(loc="lower left")
plt.savefig("loss_cifar.png")

autoencoder.save_weights('CNN_model.h5')

pre_x = autoencoder.predict(test_X)
pre_x_fashion = autoencoder.predict(x_fashion)

# x_fashion=x_fashion.reshape(len(x_fashion),28,28)
# pre_x = pre_x.reshape(len(test_X), 28, 28);
# pre_x_fashion = pre_x_fashion.reshape(len(x_fashion), 28, 28);

# pre_x=pre_x.reshape((pre_x.shape[0], 28, 28, 1))
# pre_x_fashion=pre_x_fashion.reshape((pre_x_fashion.shape[0], 28, 28, 1))
#COMPARE ORIGINAL
f, ax = plt.subplots(4, 1)
index = 10
index2 = 100
ax[0].imshow(test_X[index].reshape(32, 32, 3))
ax[1].imshow(pre_x[index].reshape(32, 32, 3))

ax[2].imshow(x_fashion[index2].reshape(32, 32, 3))
ax[3].imshow(pre_x_fashion[index2].reshape(32, 32, 3))
plt.savefig("cifar_cnn.png")