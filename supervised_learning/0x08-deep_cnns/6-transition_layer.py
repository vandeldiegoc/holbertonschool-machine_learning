#!/usr/bin/env python3
"""module """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """that builds a transition layer"""
    initializer = K.initializers.he_normal(seed=None)

    layer = K.layers.BatchNormalization()(X)
    layer = K.layers.Activation('relu')(layer)

    number_filt = int(nb_filters * compression)

    layer = K.layers.Conv2D(filters=number_filt,
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=initializer,
                            )(layer)

    outpt = K.layers.AveragePooling2D(pool_size=2,
                                      padding='same')(layer)

    return outpt, number_filt 
