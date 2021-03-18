#!/usr/bin/env python3
"""module """
import numpy as np


def definiteness(matrix):
    """ that calculates the definiteness of a matrix """
    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.array_equal(matrix.T, matrix):
        return None

    n, m = np.linalg.eig(matrix)

    if all(n > 0):
        return 'Positive definite'
    if all(n >= 0):
        return 'Positive semi-definite'
    if all(n < 0):
        return 'Negative definite'
    if all(n <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
