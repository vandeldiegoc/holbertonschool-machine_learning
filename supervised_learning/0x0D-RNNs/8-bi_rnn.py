#!/usr/bin/env python3
""" module """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ Performs forward propagation for a bi RNN"""
    t = X.shape[0]
    m = X.shape[1]
    h = h_0.shape[1]
    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))
    H = np.zeros((t, m, h + h))

    for i in range(t):
        if i == 0:
            Hf[i] = bi_cell.forward(h_0, X[i])
            Hb[t - 1 - i] = bi_cell.backward(h_t, X[t - 1 - i])
        else:
            Hf[i] = bi_cell.forward(Hf[i - 1], X[i])
            Hb[t - 1 - i] = bi_cell.backward(Hb[t - 1 - i + 1], X[t - 1 - i])
    for i in range(t):
        H[i] = np.concatenate((Hf[i], Hb[i]), axis=1)

    Y = bi_cell.output(H)

    return H, Y
