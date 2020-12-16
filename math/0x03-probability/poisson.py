#!/usr/bin/env python3
"""module"""


class Poisson:
    """class Poisson that represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    @staticmethod
    def factorial(n):
        """ funtion return factorial"""
        if n == 0:
            return 1
        else:
            return(n * Poisson.factorial(n-1))

    def pmf(self, k):
        """Calculates the value of the PMF """
        k = int(k)
        if k < 0:
            return(0)
        f = self.factorial(k)
        e = 2.718281828459045090795598298428
        calulate = ((self.lambtha ** k) * (e ** -self.lambtha)) / f
        return(calulate)
