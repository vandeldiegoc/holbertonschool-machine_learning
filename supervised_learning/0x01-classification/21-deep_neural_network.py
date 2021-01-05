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
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

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

    @property
    def L(self):
        """getter L1"""
        return self.__L

    @property
    def cache(self):
        """getter cache"""
        return self.__cache

    @property
    def weights(self):
        """getter weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X
        for l in range(self.__L):
            z1 = np.matmul(self.__weights['W' + str(l + 1)],
                           self.__cache['A' + str(l)])
            Z = z1 + self.__weights['b' + str(l + 1)]
            self.__cache['A' + str(l + 1)] = 1 / (1 + np.exp(-Z))

        return(self.__cache['A' + str(l + 1)], self.__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression """
        m = len(Y[0])
        cost = -1 * np.sum((Y * np.log(A)) +
                           ((1 - Y) * np.log(1.0000001 - A))) / m
        return(cost)

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        all_c = np.where(self.forward_prop(X)[0] <= 0.5, 0, 1)
        eva = self.cost(Y, self.forward_prop(X)[0])
        return(all_c, eva)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """gradient decent"""
        m = len(Y[0])
        for i in reversed(range(self.__L)):
            if i+1 == self.__L:
                dz = cache["A" + str(i + 1)] - Y
            else:
                dz = da * (cache["A" + str(i + 1)] *
                           (1 - cache["A" + str(i + 1)]))

            dw = 1 / m * np.matmul(dz, cache["A" + str(i)].T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            da = 1 / m * np.matmul(self.__weights['W' + str(i + 1)].T, dz)
            self.__weights['W' + str(i+1)] = (self.__weights['W' + str(i+1)] -
                                              (alpha * dw))
            self.__weights['b' + str(i+1)] = (self.__weights['b' + str(i+1)] -
                                              (alpha * db))
