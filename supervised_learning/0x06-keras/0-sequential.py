#!/usr/bin/env python3
"""module"""


import tensorflow.k as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """that builds a neural network with the k library"""
    model = k.Sequential()
    L2 = k.regularizers.L1L2(l2=lambtha)

    for i in range(len(layers)):
        model.add(k.layers.Dense(units=layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=L2,
                                 input_shape=(nx,)))
        if i != len(layers) - 1:
            model.add(k.layers.Dropout(1 - keep_prob))

    return model
