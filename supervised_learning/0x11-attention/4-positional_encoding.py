#!/usr/bin/env python3
"""modile"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """ calculates the positional
        encoding for a transformer: """
    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :],
                            dm)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding[0]


def get_angles(pos, i, d_model):
    """calculated the angle  """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
