#!/usr/bin/env python3
""" Convert array of indices to one-hot encoded numpy array """
import numpy as np


def one_hot_encode(Y, classes):
    """funtion one_hot_encode """
    try:
        res = np.eye(classes)[np.array(Y).reshape(-1)]
        return res.reshape(list(Y.shape)+[classes]).T
    except Exception:
        return(None)