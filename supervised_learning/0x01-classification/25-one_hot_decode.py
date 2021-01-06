#!/usr/bin/env python3
""" hat converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """ one_hot_decode """
    try:
        class_labels = np.argmax(one_hot, axis=0)
        if type(class_labels) != np.ndarray:
            return(None)
        return(class_labels)
    except Exception:
        return(None)
