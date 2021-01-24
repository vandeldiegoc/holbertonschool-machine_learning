#!/usr/bin/env python3
""" module """

import numpy as np


def normalize(X, m, s):
    """ that normalizes (standardizes) a matrix:"""
    nomal = (X - m) / s
    return(nomal)
