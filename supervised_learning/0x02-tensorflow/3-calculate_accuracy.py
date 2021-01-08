#!/usr/bin/env python3
"""funtion that calculates the accuracy of a prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Returns: a tensor containing the decimal accuracy of the predictions"""
    y_label = tf.argmax(y, 1)
    pred = tf.argmax(y_pred, 1)
    equal = tf.equal(pred, y_label)
    return tf.reduce_mean(tf.cast(equal, tf.float32))
