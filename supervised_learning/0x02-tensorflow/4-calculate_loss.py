#!/usr/bin/env python3
"""funtion that calculates the accuracy of a prediction"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ def calculate_loss(y, y_pred): """
    return(tf.losses.softmax_cross_entropy(y, y_pred))
