#!/usr/bin/env python3
"""module"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """builds the DenseNet-121 architecture"""
    X = K.Input(shape=(224, 224, 3))
    layer = K.layers.BatchNormalization()(X)
    act = K.layers.Activation("relu")(layer)
    conv = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                           kernel_initializer="he_normal",
                           padding="same")(act)
    pooli = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(conv)
    dense1, nb_filters = dense_block(pooli, 64, growth_rate, 6)
    trl1, nb_filters = transition_layer(dense1, nb_filters, compression)
    dense2, nb_filters = dense_block(trl1, nb_filters, growth_rate, 12)
    trl2, nb_filters = transition_layer(dense2, nb_filters, compression)
    dense3, nb_filters = dense_block(trl2, nb_filters, growth_rate, 24)
    trl3, nb_filters = transition_layer(dense3, nb_filters, compression)
    dense4, nb_filters = dense_block(trl3, nb_filters, growth_rate, 16)
    avg1 = K.layers.AveragePooling2D(pool_size=7)(dense4)
    output = K.layers.Dense(1000, activation="softmax")(avg1)
    model = K.models.Model(inputs=X, outputs=output)
    return(model)
