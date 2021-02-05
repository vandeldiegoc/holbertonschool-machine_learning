#!/usr/bin/env python3
"""module """
import tensorflow.keras as K


def lenet5(x, y):
    """hat builds a modified version
       of the LeNet-5 architecture using keras """
    initializer = K.initializers.he_normal(seed=None)

    my_layer = K.layers.Conv2D(filters=6,
                               kernel_size=(5, 5),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(x)

    my_layer = K.layers.MaxPool2D(pool_size=(2, 2),
                                  strides=(2, 2))(my_layer)

    my_layer = K.layers.Conv2D(filters=16,
                               kernel_size=(5, 5),
                               padding='valid',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(my_layer)

    my_layer = K.layers.MaxPool2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  )(my_layer)

    my_layer = K.layers.Flatten()(my_layer)

    my_layer = K.layers.Dense(units=120,
                              activation='relu',
                              kernel_initializer=initializer,
                              )(my_layer)

    my_layer = K.layers.Dense(units=84,
                              activation='relu',
                              kernel_initializer=initializer,
                              )(my_layer)

    my_layer = K.layers.Dense(units=10,
                              activation='softmax',
                              kernel_initializer=initializer,
                              )(my_layer)

    network = K.Model(inputs=x, outputs=my_layer)

    network.compile(optimizer=K.optimizers.Adam(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return network
