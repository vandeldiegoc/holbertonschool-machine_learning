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
