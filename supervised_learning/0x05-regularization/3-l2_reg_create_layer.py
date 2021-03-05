#!/usr/bin/env python3
"""module"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ creates a tensorflow layer
        that includes L2 regularization """
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    model = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=reg)

    return model(prev)
