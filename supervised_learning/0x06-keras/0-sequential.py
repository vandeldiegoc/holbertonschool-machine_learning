#!/usr/bin/env python3
"""module"""


import tensorflow as tf
import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """that builds a neural network with the Keras library"""
    model = keras.Sequential()
    L2 = keras.regularizers.L1L2(l2=lambtha)

    for i in range(len(layers)):
        model.add(keras.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=L2,
                                     input_shape=(nx,)))
        if i != len(layers) - 1:
            model.add(keras.layers.Dropout(1 - keep_prob))

    return model
