#!/usr/bin/env python3
""" module """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """that builds an inception block as described in
       Going Deeper with Convolutions (2014) """
    F1, F3R, F3, F5R, F5, FPP = filters

    initializer = K.initializers.he_normal(seed=None)

    layer = K.layers.Conv2D(filters=F1,
                            kernel_size=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=initializer,
                            )(A_prev)

    layer1 = K.layers.Conv2D(filters=F3R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer=initializer,
                             )(A_prev)

    layer1 = K.layers.Conv2D(filters=F3,
                             kernel_size=(3, 3),
                             padding='same',
                             activation='relu',
                             kernel_initializer=initializer,
                             )(layer1)

    layer2 = K.layers.Conv2D(filters=F5R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer=initializer,
                             )(A_prev)

    layer2 = K.layers.Conv2D(filters=F5,
                             kernel_size=(5, 5),
                             padding='same',
                             activation='relu',
                             kernel_initializer=initializer,
                             )(layer2)

    layer3 = K.layers.MaxPool2D(pool_size=(3, 3),
                                padding='same',
                                strides=(1, 1))(A_prev)

    layer3 = K.layers.Conv2D(filters=FPP,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer=initializer,
                             )(layer3)

    output = K.layers.concatenate([layer, layer1, layer2, layer3])

    return output
