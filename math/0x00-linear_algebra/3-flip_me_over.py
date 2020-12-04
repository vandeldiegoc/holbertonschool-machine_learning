#!/usr/bin/env python3
""" transpose of a 2D matrix """


def matrix_transpose(matrix):
    """ cat mat 2d"""
    new_array = []
    for inx in range(len(matrix[0])):
        temp = []
        for j in matrix:
            temp.append(j[inx])
        new_array.append(temp)
    return(new_array)
