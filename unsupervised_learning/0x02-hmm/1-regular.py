#!/usr/bin/env python3
""" module"""

import numpy as np


def regular(P):
    """ that determines the probability of a markov
        chain being in a particular state after a
        specified number of iteration """
    if not isinstance(P, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if ((P > 0).all()):
        dim = P.shape[0]
        q = (P-np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q, ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        return np.linalg.solve(QTQ, bQT)
