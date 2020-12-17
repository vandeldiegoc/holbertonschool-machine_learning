#!/usr/bin/env python3
"""module"""


class Binomial:
    """  represents a binomial distribution: """
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            v_1 = 0
            for x in range(len(data)):
                v_1 += (data[x] - mean) ** 2
            v_1 = v_1 / len(data)
            v = 1 - (v_1 / mean)
            self.n = round(mean / v)
            self.p = mean / self.n

    @staticmethod
    def f(n):
        """ funtion return factorial"""
        if n == 0:
            return 1
        else:
            return(n * Binomial.f(n-1))

    def pmf(self, k):
        """ Calculates the value of the PMF"""
        k = int(k)
        if k < 0:
            return(0)
        q = 1 - self.p
        pmf_1 = self.f(self.n) / (self.f(self.n-k) * self.f(k))
        pmf = (pmf_1 * (self.p ** k)) * (q ** (self.n - k))
        return(pmf)

    def cdf(self, k):
        """ Calculates the value of the CDF """
        k = int(k)
        if k < 0:
            return 0
        v = 0
        for x in range(k + 1):
            v += self.pmf(x)
        return v
