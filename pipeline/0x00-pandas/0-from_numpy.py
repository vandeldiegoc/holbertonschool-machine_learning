#!/usr/bin/env python3
""" module """
import pandas as pd


def from_numpy(array):
    """ Convert an np array to a datafram"""
    alphabet_string = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                       'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                       'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    rang = (array.shape[1])
    return pd.DataFrame(array, columns=alphabet_string[: rang])
