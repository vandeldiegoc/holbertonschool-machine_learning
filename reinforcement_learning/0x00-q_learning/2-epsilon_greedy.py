#!/usr/bin/env python3
"""module """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """that uses epsilon-greedy to
       determine the next action:"""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state, :])
