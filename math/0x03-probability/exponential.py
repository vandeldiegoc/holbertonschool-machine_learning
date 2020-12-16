#!/usr/bin/env python3
"""module"""


class Exponential:
    """represents an exponential distribution:"""
    euler = 2.7182818285

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
            self.lambtha = 1 / float(sum(data) / len(data))

    def pdf(self, x):
        """Calculates the value of the PMF """
        if x < 0:
            return(0)
        pdf = self.lambtha * (self.euler ** ((-self.lambtha) * x))
        return(pdf)
