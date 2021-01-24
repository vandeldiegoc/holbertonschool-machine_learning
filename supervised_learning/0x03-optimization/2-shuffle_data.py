#!/usr/bin/env python3
""" module """

import numpy as np


def shuffle_data(X, Y):
    """ that shuffles the data"""
    x = np.random.permutation(X)
    y = np.random.permutation(Y)
    return x, y
