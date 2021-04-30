#!/usr/bin/env python3
""" module """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ Performs forward propagation for a simple RNN:"""
    T, m, i = X.shape
    o = rnn_cell.by.shape[1]
    h = h_0.shape[1]
    H = np.zeros((T + 1, m, h))
    Y = np.zeros((T, m, o))

    H[0] = h_0
    for t in range(1, T + 1):
        H[t], Y[t - 1] = rnn_cell.forward(H[t - 1], X[t - 1])

    return H, Y
