#!/usr/bin/env python3
"""module """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ function that calculates the expectation
        step in the EM algorithm for a GMM """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None
    n, d = X.shape
    if pi.shape[0] > n:
        return None, None
    k = pi.shape[0]
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None
    g = np.zeros((k, n))
    for i in range(k):
        PDF = pdf(X, m[i], S[i])
        g[i] = pi[i] * PDF
    sum_gis = np.sum(g, axis=0, keepdims=True)
    g /= sum_gis
    lkhd = np.sum(np.log(sum_gis))
    return g, lkhd
