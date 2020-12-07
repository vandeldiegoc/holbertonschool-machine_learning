#!/usr/bin/env python3
""" module """


def np_elementwise(mat1, mat2):
    """ funtion """
    add = mat1+mat2
    sub = mat1-mat2
    mul = mat1*mat2
    div = mat1/mat2
    return(add, sub, mul, div)