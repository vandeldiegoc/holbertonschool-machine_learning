#!/usr/bin/env python3
"""module"""
import numpy as np


class DeepNeuralNetwork:
    """
    class DeepNeuralNetwork that defines
    a deep neural network performing binary
    classification
    """
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integerser")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for p_i in range(len(layers)):
            if (type(layers[p_i]) is not int or layers[p_i] < 1):
                raise TypeError("layers must be a list of positive integers")
            key_W = "W{}".format(p_i + 1)
            key_b = "b{}".format(p_i + 1)
            if p_i == 0:
                w = np.random.randn(layers[p_i], nx) * np.sqrt(2 / nx)
                self.weights[key_W] = w
            else:
                heteal_2 = np.sqrt(2 / layers[p_i - 1])
                w = np.random.randn(layers[p_i], layers[p_i - 1]) * heteal_2
                self.weights[key_W] = w
            b = np.zeros((layers[p_i], 1))
            self.weights[key_b] = b
