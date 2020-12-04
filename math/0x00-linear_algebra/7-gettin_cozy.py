#!/usr/bin/env python3
""" cat matrices """


def cat_matrices2D(mat1, mat2, axis=0):
    """ cat matrices """
    new_array = []
    temp = []
    if axis == 0 and (len(mat1[0]) == len(mat2[0])):
        return(mat1 + mat2)
    elif axis == 1 and (len(mat1) == len(mat2)):
        for i in range(len(mat1)):
            lis = mat1[i] + mat2[i]
            new_array.append(lis)
        return(new_array)
