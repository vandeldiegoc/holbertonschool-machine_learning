#!/usr/bin/env python3
""" module"""

import numpy as np


def absorbing(P):
    """that determines if a markov chain is absorbing"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None
    if np.all(np.diag(P) != 1):
        return False
    if np.all(np.diag(P) == 1):
        return True
    for i in range(n):
        if np.any(P[i, :] == 1):
            continue
        break
    II = P[:i, :i]
    Id = np.identity(n - i)
    R = P[i:, :i]
    Q = P[i:, i:]
    try:
        F = np.linalg.inv(Id - Q)
    except Exception:
        return False
    FR = np.matmul(F, R)
    Pbar = np.zeros((n, n))
    Pbar[:i, :i] = P[:i, :i]
    Pbar[i:, :i] = FR
    Qbar = Pbar[i:, i:]
    if np.all(Qbar == 0):
        return True
