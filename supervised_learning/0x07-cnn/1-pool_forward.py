#!/usr/bin/env python3
""" module """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """that performs forward propagation
       over a pooling layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    outp_h = int(((h_prev - kh) / sh) + 1)
    outp_w = int(((w_prev - kw) / sw) + 1)
    output = np.zeros((m, outp_h, outp_w, c_prev))
    for x in range(outp_w):
        for y in range(outp_h):
            if mode == 'max':
                output[:, y, x, :] = \
                    np.max(A_prev[:,
                           y * sh: y * sh + kh,
                           x * sw: x * sw + kw], axis=(1, 2))
            if mode == 'avg':
                output[:, y, x, :] = \
                    np.mean(A_prev[:,
                            y * sh: y * sh + kh,
                            x * sw: x * sw + kw], axis=(1, 2))
    return output
