#!/usr/bin/env python3
"""funtion that creates the training operation for the network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Returns: an operation that trains the network using gradient descent"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
