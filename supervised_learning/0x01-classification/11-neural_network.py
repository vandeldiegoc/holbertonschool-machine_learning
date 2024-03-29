#!/usr/bin/env python3
"""module"""
import numpy as np


class NeuralNetwork:
    """NeuralNetwork"""
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes <= 0:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter W1"""
        return self.__W1

    @property
    def b1(self):
        """getter b1"""
        return self.__b1

    @property
    def A1(self):
        """getter A1"""
        return self.__A1

    @property
    def W2(self):
        """getter W2"""
        return self.__W2

    @property
    def b2(self):
        """getter b2"""
        return self.__b2

    @property
    def A2(self):
        """getter A2"""
        return self.__A2

    def forward_prop(self, X):
        """sigmoid activation function """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1+np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1+np.exp(-z2))
        return(self.__A1, self.__A2)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression """
        m = len(Y[0])
        cost = -1 * np.sum((Y * np.log(A)) +
                           ((1 - Y) * np.log(1.0000001 - A))) / m
        return(cost)
