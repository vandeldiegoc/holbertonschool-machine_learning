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
        """getter w"""
        return(self.__W)

    @property
    def b(self):
        """getter b"""
        return(self.__b)

    @property
    def A(self):
        """getter A"""
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

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        a = self.forward_prop(X)
        all_c = np.where(a <= 0.5, 0, 1)
        eva = self.cost(Y, a)
        return(all_c, eva)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        m = len(X[0])
        dz = A - Y
        dw = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W = self.__W - (alpha * dw).T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """train neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
