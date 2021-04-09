#!/usr/bin/env python3
""" 4-viterbi.py """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """that that calculates the most likely sequence of hidden states
    for a hidden markov model"""
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    if not np.isclose(np.sum(Initial, axis=0), [1])[0]:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None
    if not np.isclose(np.sum(Transition, axis=1),
                      np.ones(Initial.shape[0])).all():
        return None, None
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.isclose(np.sum(Emission, axis=1),
                      np.ones(Emission.shape[0])).all():
        return None, None

    N = Initial.shape[0]
    T = Observation.shape[0]
    V = np.zeros((N, T))
    V[:, 0] = Initial.T * Emission[:, Observation[0]]
    B = np.zeros((N, T))

    for j in range(1, T):
        for i in range(N):
            temp = Emission[i, Observation[j]] * Transition[:, i] * V[:, j - 1]
            V[i, j] = np.max(temp, axis=0)
            B[i, j] = np.argmax(temp, axis=0)
    P = np.max(V[:, T - 1])
    S = np.argmax(V[:, T - 1])
    path = [S]
    for j in range(T - 1, 0, -1):
        S = int(B[S, j])
        path.append(S)
    path = path[:: -1]
    return path, P
