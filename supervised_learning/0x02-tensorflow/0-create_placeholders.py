#!/usr/bin/env python3
"""function that return placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """create_placeholder"""
    x = tf.placeholder(name='x', shape=(None, nx), dtype=tf.float32)
    y = tf.placeholder(name='y', shape=(None, classes), dtype=tf.float32)
    return(x, y)
