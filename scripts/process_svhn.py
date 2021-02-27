#!/usr/bin/env python3

import pickle

import tensorflow as tf

from keras import Model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import scipy.io as sio

from sys import exit

X_train = []
y_train = []

for f in ["train_32x32.mat", "test_32x32.mat", "extra_32x32.mat"]:
    p = "../data/svhn/" + f
    mat = sio.loadmat(p)

    X = mat.get("X").transpose(3, 0, 1, 2)
    y = mat.get("y")

    if f.startswith('test'):
        X_test = X
        y_test = to_categorical(y - 1)
    else:
        X_train.append(X)
        y_train.append(y)

X_train = np.concatenate(X_train)
y_train = to_categorical(np.concatenate(y_train) - 1)

model = Sequential()
model.add(Input(shape=(32, 32, 3)))

model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Lambda(tf.nn.local_response_normalization))
model.add(Dropout(0.2))

model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation="relu", name="embedding"))
model.add(Dense(10, activation="softmax"))

opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=100, epochs=1, validation_data=(X_test, y_test))
_, acc = model.evaluate(X_test, y_test)

print("accuracy =", acc)

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)
intermediate_output = intermediate_layer_model.predict(np.concatenate([X_train, X_train]))

df = pd.DataFrame(data=intermediate_output, columns=[f"x{i+1}" for i in range(intermediate_output.shape[1])])
df.to_csv("svhn_d_128.csv", index=None)
