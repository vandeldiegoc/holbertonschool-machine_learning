#!/usr/bin/env python3
"""module"""

import tensorflow.keras as k


def one_hot(labels, classes=None):
    """  that converts a label vector into a one-hot matrix:"""
    return k.utils.to_categorical(labels, classes)
