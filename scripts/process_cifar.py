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

import numpy as np
import pandas as pd

from sys import exit

labels = []
arys = []
for f in [f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 5)]:
    with open(f, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')

    data = d[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    print(data[0, :, :, :].shape)

    arys.append(data)
    labels.append(d[b'labels'])

train_X = np.concatenate(arys) / 255.0
train_y = to_categorical(np.concatenate(labels))

with open('cifar-10-batches-py/test_batch', 'rb') as f:
    d = pickle.load(f, encoding='bytes')
    test_X = d[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0
    test_y = to_categorical(d[b'labels'])

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

history = model.fit(train_X, train_y, batch_size=100, epochs=100, validation_data=(test_X, test_y))
_, acc = model.evaluate(test_X, test_y)

print("accuracy =", acc)

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)
intermediate_output = intermediate_layer_model.predict(np.concatenate([train_X, test_X]))

df = pd.DataFrame(data=intermediate_output, columns=[f"x{i+1}" for i in range(intermediate_output.shape[1])])
df.to_csv("cifar_128d.csv", index=None)
