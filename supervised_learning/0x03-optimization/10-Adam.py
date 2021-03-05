#!/usr/bin/env python3
"""Contains the create_RMSProp_op function"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ creates the training operation
        for a neural network in tensorflow
        using the RMSProp optimization """
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return optimizer.minimize(loss)
