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


def closest_centroid(points, centroids):
    """returns an array containing the idx to the nearest centroid"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def kmeans(X, k, iterations=1000):
    """ kmeans algorithm"""
    cent = initialize(X, k)
    if cent is None:
        return None, None
    if not isinstance(iterations, int):
        return None, None
    if iterations <= 0:
        return None, None
    closest = closest_centroid(X, cent)
    for i in range(iterations):
        cp = np.copy(cent)
        for j in range(k):
            if X[np.where(closest == j)].size == 0:
                cent[j] = initialize(X, 1)
            else:
                cent[j] = X[np.where(closest == j)].mean(axis=0)
        closest = closest_centroid(X, cent)
        if np.array_equal(cp, cent):
            break

    return cent, closest
