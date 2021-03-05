#!/usr/bin/env python3
"""module"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ update variable using RMSProp"""
    sdw = beta2 * s + (1 - beta2) * (grad ** 2)
    w = var - alpha * grad / (sdw ** 0.5 + epsilon)
    return w, sdw
