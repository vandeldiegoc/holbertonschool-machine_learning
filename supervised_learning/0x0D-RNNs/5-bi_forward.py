#!/usr/bin/env python3
""" module """
import numpy as np


class BidirectionalCell:
    """bidirectional cell RNN"""
    def __init__(self, i, h, o):
        """Class constructor"""
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
        """ feed forward algo for RNN"""
        conc = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(conc.dot(self.Whf) + self.bhf)
        return h_next
