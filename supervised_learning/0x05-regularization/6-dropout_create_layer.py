#!/usr/bin/env python3
"""module"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates a layer of a neural network using dropount"""
    reg = tf.layers.Dropout(keep_prob)
    w = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=w,
        kernel_regularizer=reg,
        activation=activation)
    return layer(prev)
