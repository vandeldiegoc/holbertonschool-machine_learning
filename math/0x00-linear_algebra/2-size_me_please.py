#!/usr/bin/env python3
""" calculates the shape of a matrix """


def matrix_shape(matrix):
    """ count matrix"""
    if matrix is None:
        return None
    if type(matrix) == list:
        if type(matrix[0]) == list:
            return [len(matrix)] + matrix_shape(matrix[0])
        return [len(matrix)]
    else:
        return []
