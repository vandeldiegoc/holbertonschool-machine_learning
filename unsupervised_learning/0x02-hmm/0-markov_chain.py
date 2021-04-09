#!/usr/bin/env python3
""" module"""

import numpy as np


def markov_chain(P, s, t=1):
    """ that determines the probability of a markov
        chain being in a particular state after a
        specified number of iteration """
    if t < 0 or not isinstance(t, int):
        return None
    if not isinstance(P, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if len(s.shape) != 2 or s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(s, np.ndarray):
        return None
    if (P < 0).any():
        return None
    if not isinstance(t, int):
        return None
    for i in range(t):
        s = np.matmul(s, P)
        return(s)
