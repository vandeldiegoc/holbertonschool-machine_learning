#!/usr/bin/env python3
"""module"""
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ that builds a neural network with the k library """
    L2 = k.regularizers.L1L2(l2=lambtha)

    inputs = k.Input(shape=(nx,))

    layer = k.layers.Dense(units=layers[0],
                           activation=activations[0],
                           kernel_regularizer=L2,
                           input_shape=(nx,))(inputs)

    for i in range(1, len(layers)):

        layer = k.layers.Dropout(1 - keep_prob)(layer)
        layer = k.layers.Dense(units=layers[i],
                               activation=activations[i],
                               kernel_regularizer=L2)(layer)
    model = k.Model(inputs=inputs, outputs=layer)
    return model
