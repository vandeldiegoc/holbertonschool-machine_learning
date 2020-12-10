#!/usr/bin/env python3
"""module """


def summation_i_squared(n):
    """ Sums of powers """
    if type(n) != int or n <= 0:
        return None
    m = n * (n + 1)*(2 * n + 1) / 6
    return int(m)
