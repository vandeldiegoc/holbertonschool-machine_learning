#!/usr/bin/env python3
"""module"""


class Exponential:
    """represents a normal distribution:"""
    e = 2.7182818285

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
        """Calculates the value of the PDF """
        if x < 0:
            return(0)
        pdf = self.lambtha * (self.e ** ((-self.lambtha) * x))
        return(pdf)

    def cdf(self, x):
        """Calculates the value of the PMF """
        if x < 0:
            return(0)
        cdf = 1 - (self.e ** ((-self.lambtha) * x))
        return(cdf)
