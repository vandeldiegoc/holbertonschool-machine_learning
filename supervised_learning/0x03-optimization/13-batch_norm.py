#!/usr/bin/env python3
"""module"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ batch normalizatoin fucntion"""
    m = Z.shape[0]
    u = np.sum(Z, axis=0) / m
    g = np.sum((Z - u) ** 2, axis=0) / m
    z_norm = (Z - u) / ((g + epsilon) ** 0.5)
    return gamma * z_norm + beta
