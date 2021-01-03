#!/usr/bin/env python3
import numpy as np


class Neuron:
    """
    class that work like a neurron
    """
    e = 2.7182818285

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 0:
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
