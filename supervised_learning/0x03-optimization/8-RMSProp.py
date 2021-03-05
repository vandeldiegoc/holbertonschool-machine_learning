#!/usr/bin/env python3
"""modue"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """that creates the training operation for a neural
       network in tensorflow using the RMSProp"""
    return tf.train.RMSPropOptimizer(
        alpha, beta2, epsilon=epsilon).minimize(loss)
