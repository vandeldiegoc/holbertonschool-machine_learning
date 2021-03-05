#!/usr/bin/env python3
""" momentum upgraded"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """creates the training operation for a neural network in tensorflow"""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)