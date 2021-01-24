#!/usr/bin/env python3
""" module """

import numpy as np


def normalization_constants(X):
    """[take the input and calculate the mean,
        and the standard deviation]

    Args:
        X ([numpy.ndarray]): [numpy.ndarray of shape (m, nx)
                              to normalize]

    Returns:
        [type]: [the mean and standard deviation
                 of each feature, respectivel]
    """
    mean = np.mean(X, axis=0)
    st = np.std(X, axis=0)
    return mean, st
