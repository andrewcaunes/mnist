# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import tensorflow as tf
import numpy as np

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

print("X_train ", X_train.shape)
print("Y_train ", Y_train.shape)
print("X_test ", X_test.shape)
print("Y_test", Y_test.shape)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss = tf.keras.losses.sparse_categorical_crossentropy()
optimizer =