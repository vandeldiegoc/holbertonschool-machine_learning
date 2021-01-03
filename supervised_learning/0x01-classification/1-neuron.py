#!/usr/bin/env python3
""" module """
import numpy as np


class Neuron:
    """class that work like a neurron """
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
