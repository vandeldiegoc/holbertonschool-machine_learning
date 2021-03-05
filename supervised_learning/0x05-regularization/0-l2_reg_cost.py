#!/usr/bin/env python3
""" module """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ l2_reg_cost - calculates the cost of a neural network"""
    layers = 0
    for i in range(1, L + 1):
        k = 'W{}'.format(i)
        layers += np.linalg.norm(weights[k])

    L2 = lambtha * layers / (2 * m)

    return (cost + L2)
