#!/usr/bin/env python3
""" module """
import numpy as np


class LSTMCell:
    """class LSTMCell"""

    def __init__(self, i, h, o):
        """constructor"""

        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """function that performs forward propagation for one time step"""
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        forget_gate = self.sigmoid(np.matmul(cell_input, self.Wf) + self.bf)
        update_gate = self.sigmoid(np.matmul(cell_input, self.Wu) + self.bu)
        c_inter = np.tanh(np.matmul(cell_input, self.Wc) + self.bc)
        c_next = c_prev * forget_gate + update_gate * c_inter
        output_gate = self.sigmoid(np.matmul(cell_input, self.Wo) + self.bo)
        h_next = output_gate * np.tanh(c_next)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, c_next, y

    def softmax(self, Y):
        """softmax activation function"""
        return np.exp(Y) / (np.sum(np.exp(Y), axis=1, keepdims=True))

    def sigmoid(self, Y):
        """sigmoid activation function"""
        return 1 / (1 + np.exp(-Y))

