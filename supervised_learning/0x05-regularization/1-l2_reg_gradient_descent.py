#!/usr/bin/env python3
""" module """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural network
    using gradient descent with L2 regularization"""
    m = Y.shape[1]
    weis = weights.copy()
    for i in range(L - 1, -1, -1):
        prev_A = cache['A' + str(i)]
        A = cache['A' + str(i + 1)]
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        if i == L - 1:
            dz = A - Y
        else:
            dz = np.matmul(weis['W' + str(i + 2)].T, dz) * (1 - (A * A))
        dW = np.matmul(dz, prev_A.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W' + str(i + 1)] = weights["W" + str(i + 1)] - \
            alpha * lambtha * weights['W' + str(i + 1)] / m - alpha * dW
        weights['b' + str(i + 1)] = weights["b" + str(i + 1)] - (alpha * db)
