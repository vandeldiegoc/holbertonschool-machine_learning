#!/usr/bin/env python3
"""module"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of a neural network
       with Dropout regularization using
       gradient descent """
    weights_temp = weights.copy()
    m = Y.shape[1]

    for i in reversed(range(L)):
        if i == L - 1:
            dZ = cache['A{}'.format(i + 1)] - Y

        else:
            dZa = np.matmul(weights_temp['W{}'.format(i + 2)].T, dZ)
            dZb = 1 - cache['A{}'.format(i + 1)] ** 2
            dZ = dZa * dZb
            dZ *= cache["D{}".format(i + 1)]
            dZ /= keep_prob

        dW = (np.matmul(dZ, cache['A{}'.format(i)].T)) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W{}'.format(i + 1)] = \
            weights_temp['W{}'.format(i + 1)] \
            - (alpha * dW)
        weights['b{}'.format(i + 1)] = \
            weights_temp['b{}'.format(i + 1)] \
            - (alpha * db)
