#!/usr/bin/env python3
""" module """
import numpy as np


def variance(X, C):
    """  that calculates the total intra-cluster variance for a data sets"""
    if not isinstance(X, np.ndarray):
        return None
    if len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray):
        return None
    if len(C.shape) != 2 or C.shape[1] != X.shape[1]:
        return None
    dist = np.sqrt((X[np.newaxis] - C[:, np.newaxis])**2).sum(axis=2)\
        .argmin(axis=0)
    v = np.linalg.norm(X - C[dist]) ** 2
    return v
