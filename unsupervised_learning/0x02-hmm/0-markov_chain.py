#!/usr/bin/env python3
""" module"""

import numpy as np


def markov_chain(P, s, t=1):
    """ that determines the probability of a markov
        chain being in a particular state after a
        specified number of iteration """
    if (not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray)):
        return None

    if (not isinstance(t, int)):
        return None

    if ((P.ndim != 2) or (s.ndim != 2) or (t < 1)):
        return None

    n = P.shape[0]
    if (P.shape != (n, n)) or (s.shape != (1, n)):
        return None
    for i in range(t):
        s = np.matmul(s, P)
    return(s)
