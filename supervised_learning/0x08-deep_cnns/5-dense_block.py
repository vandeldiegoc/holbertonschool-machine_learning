#!/usr/bin/env python3
"""module"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds a dense block """
    initializer = K.initializers.he_normal(seed=None)

    for i in range(layers):
        my_layer = K.layers.BatchNormalization()(X)
        my_layer = K.layers.Activation('relu')(my_layer)

        my_layer = K.layers.Conv2D(filters=4*growth_rate,
                                   kernel_size=1,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   )(my_layer)

        my_layer = K.layers.BatchNormalization()(my_layer)
        my_layer = K.layers.Activation('relu')(my_layer)

        my_layer = K.layers.Conv2D(filters=growth_rate,
                                   kernel_size=3,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   )(my_layer)

        X = K.layers.concatenate([X, my_layer])
        nb_filters += growth_rate

    return X, nb_filters
