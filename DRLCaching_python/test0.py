import tensorflow.keras as tf
import numpy as np

layer_1 = tf.layers.Dense(units=10, activation=tf.activations.relu)
x = np.array([[1, 2, 3],
              [4, 5, 6]])
a = layer_1(x)
print(a)
