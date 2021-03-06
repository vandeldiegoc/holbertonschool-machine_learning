#!/usr/bin/env python3
"""module"""
import numpy as np


class Neuron:
    """class that work like a neurron"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return(self.__W)

    @property
    def b(self):
        return(self.__b)

    @property
    def A(self):
        return(self.__A)

    def forward_prop(self, X):
        """sigmoid activation function """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1+np.exp(-z))
        return(self.__A)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression """
        m = len(Y[0])
        cost = -1 * np.sum((Y * np.log(A)) +
                           ((1 - Y) * np.log(1.0000001 - A))) / m
        return(cost)
