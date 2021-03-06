#!/usr/bin/env python3
"""module"""


def np_slice(matrix, axes={}):
    """slices a matrix along specific axes"""
    ax = []
    i = 0
    for number, element in axes.items():
        while i != number:
            i += 1
            ax.append(slice(None))
        i += 1
        ax.append(slice(*element))
    ax = tuple(ax)
    return matrix[ax]
