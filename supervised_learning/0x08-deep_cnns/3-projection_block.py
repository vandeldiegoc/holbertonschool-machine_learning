#!/usr/bin/env python3
""" module"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """that builds an identity block """
    F11, F3, F12 = filters

    initializer = K.initializers.he_normal(seed=None)

    layer = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            strides=(s, s),
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

    a_prev = K.layers.Conv2D(filters=F12,
                             kernel_size=(1, 1),
                             strides=(s, s),
                             padding='same',
                             kernel_initializer=initializer,
                             )(A_prev)

    a_prev = K.layers.BatchNormalization(axis=3)(a_prev)

    output = K.layers.Add()([layer, a_prev])

    output = K.layers.Activation('relu')(output)

    return output
