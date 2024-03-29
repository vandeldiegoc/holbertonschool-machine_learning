#!/usr/bin/env python3
""" that adds two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ add matrices """
    if len(mat1) != len(mat2):
        return(None)
    new_array = []
    for inx in range(len(mat1)):
        temp = []
        if len(mat1[inx]) != len(mat2[inx]):
            return(None) 
        for j in range(len(mat1[inx])):
            temp.append(mat1[inx][j] + mat2[inx][j])
        new_array.append(temp)
    return(new_array)
