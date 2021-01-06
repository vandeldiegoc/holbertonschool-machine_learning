#!/usr/bin/env python3
"""module"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        self.forward_prop(X)
        all_c = np.where(self.__A2 <= 0.5, 0, 1)
        eva = self.cost(Y, self.__A2)
        return(all_c, eva)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates of gradient descent neurons"""
        m = len(A2[0])
        dZ2 = A2 - Y
        dW2 = 1 / m * np.matmul(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.matmul(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = 1 / m * np.matmul(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dW2)
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """train neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        Cost_after = []
        iteration = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if i % step == 0 or i == iterations:
                Cost_after.append(self.cost(Y, self.__A2))
                iteration.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".
                          format(i, self.cost(Y, self.cost(Y, self.__A2)))

        if graph is True:
            if graph:
                plt.plot(iteration, Cost_after)
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.title('Training cost')
                plt.show()
        return self.evaluate(X, Y)
