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
        print('p', P)
        dim = P.shape[0]
        print(dim)
        q = (P-np.eye(dim))
        print('q', q)
        ones = np.ones(dim)
        print('one', ones)
        q = np.c_[q, ones]
        print('q', q)
        QTQ = np.dot(q, q.T)
        print('QTQ', QTQ)
        bQT = np.ones(dim)
        print('bQt', bQT)
        return np.linalg.solve(QTQ, bQT)
