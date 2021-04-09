#!/usr/bin/env python3
""" module """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ that performs the backward algorithm for
        a hidden markov model """
    if not isinstance(Observation, np.ndarray):
        return None, None
    if len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if not isinstance(Emission, np.ndarray):
        return None, None
    if len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray):
        return None, None
    if Transition.shape != (N, N):
        return None, None
    if not isinstance(Initial, np.ndarray):
        return None, None
    if Initial.shape != (N, 1):
        return None, None
    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))
    for t in range(T - 2, -1, -1):
        for s in range(N):
            tmp = B[:, t + 1] * Emission[:, Observation[t + 1]]
            B[s, t] = np.dot(tmp, Transition[s, :])

    P = np.zeros((N, T))
    for s in range(N):
        P[s, 0] = Initial[s] * Emission[s, Observation[0]]
    P = np.sum(np.matmul(P.T, B[:, 0]))
    return P, B
