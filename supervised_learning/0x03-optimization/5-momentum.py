#!/usr/bin/env python3
"""module"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updates a variable using th egradient descent of momentum"""
    vdw = beta1 * v + (1 - beta1) * grad
    w = var - alpha * vdw
    return w, vdw
