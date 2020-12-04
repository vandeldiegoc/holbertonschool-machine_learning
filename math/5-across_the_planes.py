#!/usr/bin/env python3
""" that adds two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ add matrices """
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    new_array = []
    for inx in range(len(mat1[0])):
        temp = []
        for j in range(len(mat1)):
            temp.append(mat1[j][inx] + mat2[j][inx])
        new_array.append(temp)
    return new_array
