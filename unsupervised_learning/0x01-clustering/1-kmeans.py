#!/usr/bin/env python3
""" module """
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0 or k > n:
        return None

    k = np.random.uniform(low=np.min(X, axis=0),
                          high=np.max(X, axis=0),
                          size=(k, d))
    return k


def kmeans(X, k, iterations=1000):
    """ that performs K-means on a dataset: """
    centroide = initialize(X, k)
    if centroide is None:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    temp = 0
    for j in range(iterations):
        dist = np.sqrt((X[np.newaxis] - centroide[:, np.newaxis])**2)\
               .sum(axis=2).argmin(axis=0)
        if j > 0:
            temp = np.copy(centroide)
        for m in range(k):
            indices = np.array([X[dist == m]])[0]
            if len(indices) == 0:
                centroide[m] = initialize(X, 1)
            else:
                centroide[m, :] = np.array(X[dist == m]).mean(axis=0)
        if np.array_equal(temp, centroide):
            return(centroide, dist)
    return (centroide, dist)