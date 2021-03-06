#!/usr/bin/env python3
"""module"""


def shape(matrix):
    """return shape in matrix"""
    shape = [len(matrix)]
    while type(matrix[0]) == list:
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape


def rec_add_matrices(mat1, mat2):
    """add two matrices"""
    sum = []
    for i in range(len(mat1)):
        if type(mat1[i]) == list:
            sum.append(rec_add_matrices(mat1[i], mat2[i]))
        else:
            sum.append(mat1[i] + mat2[i])
    return sum


def add_matrices(mat1, mat2):
    """adds two matrices"""

    if shape(mat1) != shape(mat2):
        return None

    return rec_add_matrices(mat1, mat2)