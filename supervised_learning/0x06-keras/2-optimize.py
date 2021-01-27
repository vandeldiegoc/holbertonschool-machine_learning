#!/usr/bin/env python3
"""module"""
import tensorflow.keras as k


def optimize_model(network, alpha, beta1, beta2):
    """  that sets up Adam optimization for
        a keras model with categorical crossentropy
        loss and accuracy metrics """

    opt = k.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss='categorical_crossentropy',
                    optimizer=opt, metrics=['accuracy'])
