#!/usr/bin/env python3
"""module"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """that performs the forward algorithm for
       a hidden markov mode"""
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
    F = np.zeros((N, T))
    for s in range(N):
        F[s, 0] = Initial[s] * Emission[s, Observation[0]]
    for t in range(1, T):
        for s in range(N):
            F[s, t] = np.dot(F[:, t - 1], Transition[:, s]) * \
                Emission[s, Observation[t]]
    P = np.sum(F[:, T - 1], axis=0)
    return P, F
