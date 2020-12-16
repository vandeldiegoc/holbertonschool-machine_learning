#!/usr/bin/env python3
"""module"""


class Normal:
    """ represents a normal distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))
            s = 0
            for x in range(len(data)):
                s += (data[x] - self.mean) ** 2
            s = s / len(data)
            self.stddev = float(s ** (1/2))
