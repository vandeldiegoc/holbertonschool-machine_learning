#!/usr/bin/env python3
""" module """

import numpy as np


def shuffle_data(X, Y):
    """ that shuffles the data"""
    m = len(X)
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]
