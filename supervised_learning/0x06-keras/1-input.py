#!/usr/bin/env python3
"""module"""
import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ that builds a neural network with the Keras library """
    L2 = keras.regularizers.L1L2(l2=lambtha)

    inputs = keras.Input(shape=(nx,))

    layer = keras.layers.Dense(units=layers[0],
                               activation=activations[0],
                               kernel_regularizer=L2)(inputs)
    print(len(layers))

    for i in range(1, len(layers)):

        layer = keras.layers.Dropout(1 - keep_prob)(layer)
        layer = keras.layers.Dense(units=layers[i],
                                   activation=activations[i],
                                   kernel_regularizer=L2, )(layer)
    model = keras.Model(inputs=inputs, outputs=layer)
    return model
