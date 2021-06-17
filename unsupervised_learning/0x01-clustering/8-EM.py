#!/usr/bin/env python3
"""module"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the expectation maximization for a GMM """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    n, d = X.shape
    lkhd_prev = 0
    pi, m, S = initialize(X, k)
    for i in range(iterations + 1):
        if i != 0:
            lkhd_prev = lkhd
            pi, m, S = maximization(X, g)
        g, lkhd = expectation(X, pi, m, S)
        if verbose:
            if i % 10 == 0 or i == iterations or abs(lkhd - lkhd_prev) <= tol:
                print("Log Likelihood after {} iterations: {}".
                      format(i, lkhd.round(5)))
        if abs(lkhd - lkhd_prev) <= tol:
            break

    return pi, m, S, g, lkhd
