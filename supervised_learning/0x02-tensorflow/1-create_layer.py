#!/usr/bin/env python3
"""def create_layer(prev, n, activation)"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Returns: the tensor output of the laye"""
    ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            name='layer',
                            kernel_initializer=ini,
                            activation=activation)
    return layer(prev)
