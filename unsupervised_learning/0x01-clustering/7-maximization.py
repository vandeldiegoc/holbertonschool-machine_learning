#!/usr/bin/env python3
"""module """
import numpy as np


def maximization(X, g):
    """that calculates the maximization
       step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    cluster = np.sum(g, axis=0)
    cluster = np.sum(cluster)
    if int(cluster) != X.shape[0]:
        return None, None, None

    n, d = X.shape
    k, n = g.shape

    nk = np.sum(g, axis=1)
    pi = nk / n
    mean = np.zeros((k, d))
    cov = np.zeros((k, d, d))
    for i in range(k):
        mean[i] = np.matmul(g[i], X) / nk[i]
        norm = X - mean[i]
        cov[i] = np.matmul(g[i] * norm.T, norm) / nk[i]

    return pi, mean, cov
