#!/usr/bin/env python3
"""module"""
import numpy as np


def policy(matrix, weight):
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp/np.sum(exp)
