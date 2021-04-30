#!/usr/bin/env python3
""" module """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN:"""
    length = len(rnn_cells)
    h = h_0.shape[2]
    T, m, i = X.shape
    o = rnn_cells[-1].by.shape[1]
    H = np.zeros((T + 1, length, m, h))
    Y = np.zeros((T, m, o))
    H[0, ...] = h_0

    for t in range(1, T + 1):
        H[t, 0], Y[t - 1] = rnn_cells[0].forward(H[t - 1, 0], X[t - 1])
        for i in range(1, length):
            rnn_cell = rnn_cells[i]
            H[t, i], Y[t - 1] = rnn_cells[i].forward(H[t - 1, i], H[t, i - 1])
    return H, Y
