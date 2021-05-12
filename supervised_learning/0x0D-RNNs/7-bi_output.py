#!/usr/bin/env python3
"""module """
import numpy as np


class BidirectionalCell:
    """represents a bidirectional cell RNN"""
    def __init__(self, i, h, o):
        """Class constructor:"""
        self.Whf = np.random.normal(size=(h + i, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """softmax"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ feed forward algo for bidirectional RNN"""
        conc = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(conc.dot(self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """ feed backward algo for bidirectional RNN"""
        conc = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(conc.dot(self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """function that calculates all outputs for the RNN"""
        t = H.shape[0]
        m = H.shape[1]
        Y = np.zeros((t, m, self.Wy.shape[1]))

        for i in range(t):
            Y[i] = self.softmax(np.matmul(H[i], self.Wy) + self.by)
        return Y