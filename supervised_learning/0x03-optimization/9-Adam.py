#!/usr/bin/env python3
"""module"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """updates a variable with Adam optimization"""
    vdw = beta1 * v + (1 - beta1) * grad
    sdw = beta2 * s + (1 - beta2) * grad ** 2
    vdw_c = vdw / (1 - beta1 ** t)
    sdw_c = sdw / (1 - beta2 ** t)
    w = var - alpha * vdw_c / ((sdw_c ** 0.5) + epsilon)
    return w, vdw, sdw
