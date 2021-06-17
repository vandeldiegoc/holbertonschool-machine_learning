#!/usr/bin/env python3


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ that tests for the optimum number
        of clusters by varianc"""
    if not isinstance(X, np.ndarray):
        return None, None
    if len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax < 1:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax - kmin < 1:
        return None, None
    results = []
    d_vars = []
    for i in range(kmin, kmax + 1):
        c, clss = kmeans(X, i, iterations)
        results.append((c, clss))
        d_vars.append(variance(X, c))
    for i in range(len(d_vars)):
        d_vars[i] = variance(X, results[0][0]) - d_vars[i]
    return results, d_vars
