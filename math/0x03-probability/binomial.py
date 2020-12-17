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
