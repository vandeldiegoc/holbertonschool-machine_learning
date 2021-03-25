#!/usr/bin/env python3
"""module"""
import numpy as np


def pca(X, ndim):
    """ that performs PCA on a dataset"""
    X_m = X - np.mean(X, axis=0)
    m = X.shape[0]
    u, s, vh = np.linalg.svd(X_m)
    v = np.cumsum(s) / np.sum(s)
    W = vh.T[:, :ndim]
    T = np.matmul(X_m, W)
    return T
