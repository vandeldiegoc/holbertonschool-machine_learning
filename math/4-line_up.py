#!/usr/bin/env python3
""" that adds two arrays element-wise """


def add_arrays(arr1, arr2):
    """add array """
    if len(arr1) != len(arr2):
        return None
    temp = []
    for inx in range(len(arr1)):
        a = arr1[inx] + arr2[inx]
        temp.append(a)
    return temp