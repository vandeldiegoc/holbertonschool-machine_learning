#!/usr/bin/env python3
""" module """


def poly_derivative(poly):
    """that calculates the derivative of a polynomial """
    new_list = []
    if type(poly) != list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    for x in range(1, len(poly)):
        new_list.append(x * poly[x])
    return(new_list)
