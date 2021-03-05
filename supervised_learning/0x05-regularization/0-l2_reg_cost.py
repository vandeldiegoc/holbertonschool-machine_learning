import numpy as np
""" module """


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ l2_reg_cost - calculates the cost of a neural network"""
    SUM = 0
    for i in range(L):
        SUM += np.linalg.norm(weights['W' + str(i + 1)])
    return cost + SUM * lambtha * 0.5 / m
