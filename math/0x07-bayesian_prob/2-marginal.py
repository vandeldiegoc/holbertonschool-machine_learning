#!/usr/bin/env python3
""" module """

import numpy as np


def likelihood(x, n, P):
    """ that calculates the likelihood of
        obtaining this data given various hypothetical
        probabilities of developing severe side effects"""
    db = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n-x))
    b = (P ** x) * ((1 - P) ** (n - x))
    return db * b


def intersection(x, n, P, Pr):
    """ hat calculates the intersection
        of obtaining this data with the
        various hypothetical probabilities: """
    return likelihood(x, n, P) * Pr


def marginal(x, n, P, Pr):
    """that calculates the marginal
       probability of obtaining the data"""
    if not isinstance(n, int) or n < 1:
        err = 'n must be a positive integer'
        raise ValueError(err)

    if not isinstance(x, int) or x < 0:
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)

    if x > n:
        err = 'x cannot be greater than n'
        raise ValueError(err)

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        err = 'P must be a 1D numpy.ndarray'
        raise TypeError(err)

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        err = 'Pr must be a numpy.ndarray with the same shape as P'
        raise TypeError(err)

    if np.any(P < 0) or np.any(P > 1):
        err = 'All values in P must be in the range [0, 1]'
        raise ValueError(err)

    if np.any(Pr < 0) or np.any(P > 1):
        err = 'All values in Pr must be in the range [0, 1]'
        raise ValueError(err)

    if not np.isclose([np.sum(Pr)], [1.])[0]:
        err = 'Pr must sum to 1'
        raise ValueError(err)
    return np.sum(intersection(x, n, P, Pr))
