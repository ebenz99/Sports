from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os

class_names = ['Playoffs', 'Not Playoffs']

mydir = os.getcwd()


train_data = np.load("training_data.npy")
train_labels = np.load("training_labels.npy")


test_data = np.load("testing_data.npy")
test_labels = np.load("testing_labels.npy")

train_data = train_data / train_data.max()
test_data = test_data / test_data.max()


model = keras.Sequential([
	keras.layers.Flatten(input_shape=(7,1)),
	keras.layers.Dense(64, activation=tf.nn.relu),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(10, activation=tf.nn.relu),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=10000, verbose=1)

test_loss, test_acc = model.evaluate(test_data, test_labels)

print('Test accuracy:', test_acc)