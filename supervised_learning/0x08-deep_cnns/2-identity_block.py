#!/usr/bin/env python3
""" module"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    builds a projection block
    :param A_prev:
    :param filters: tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    :param s: stride of the first convolution in both the main path
        and the shortcut connection
    :return: activated output of the identity block
    """
    F11, F3, F12 = filters

    initializer = K.initializers.he_normal(seed=None)

    layer = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=initializer,
                            )(A_prev)

    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation('relu')(layer)

    layer = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer=initializer,
                            )(layer)
    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.Activation('relu')(layer)

    layer = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=initializer,
                            )(layer)

    layer = K.layers.BatchNormalization(axis=3)(layer)

    output = K.layers.Add()([layer, A_prev])

    output = K.layers.Activation('relu')(output)

    return output
