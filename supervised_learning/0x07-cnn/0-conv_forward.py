#!/usr/bin/env python3
""" module """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional
        layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(np.ceil(((h_prev * sh) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev * sw) - sw + kw - w_prev) / 2))
    else:
        ph, pw = padding.shape
    outp_h = int(float(h_prev - kh + (2 * ph)) / float(sh)) + 1
    outp_w = int(float(w_prev - kw + (2 * pw)) / float(sw)) + 1
    images = np.pad(A_prev, [(0, 0), (ph, ph),
                             (pw, pw), (0, 0)], 'constant')
    output = np.zeros((m, outp_h, outp_w, c_new))
    for y in range(outp_h):
        for x in range(outp_w):
            for k in range(c_new):
                output[:, y, x, k] = np.sum(images[:,
                                                   y * sh: y * sh + kh,
                                                   x * sw: x *
                                                   sw + kw] *
                                            W[:, :, :, k], axis=(1, 2, 3))
    return (output + b)
