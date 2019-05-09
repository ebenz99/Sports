#tutorial from the tensorflow documentation

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


train_data = np.zeros(90,5)
counter = 0

for root,dirs,files in os.walk(mydir):
	for file in files:
		if file.endswith(".csv"):
			if counter < 3:
				with open(file, "r") as myfile:



'''


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(type(train_images[0][0][0]))
'''