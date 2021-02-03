#!/usr/bin/env python3
""" module """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional
        layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    ph, pw = 0, 0
    outp_h = int(((h_prev + 2 * ph - kh) / sh) + 1)
    outp_w = int(((w_prev + 2 * pw - kw) / sw) + 1)

    if padding == 'same':
        if kh % 2 == 0:
            ph = int((h_prev * sh + kh - h_prev) / 2)
            outp_h = int(((h_prev + 2 * ph - kh) / sh))
        else:
            ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
            outp_h = int(((h_prev + 2 * ph - kh) / sh) + 1)

        if kw % 2 == 0:
            pw = int((w_prev * sw + kw - w_prev) / 2)
            outp_w = int(((w_prev + 2 * pw - kw) / sw))
        else:
            pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
            outp_w = int(((w_prev + 2 * pw - kw) / sw) + 1)
    images = np.pad(A_prev, [(0, 0), (ph, ph),
                             (pw, pw), (0, 0)], 'constant', constant_values=0)
    output = np.zeros((m, outp_h, outp_w, c_new))
    for y in range(outp_h):
        for x in range(outp_w):
            for k in range(c_new):
                output[:, y, x, k] = np.sum(images[:,
                                                   y * sh: y * sh + kh,
                                                   x * sw: x *
                                                   sw + kw, :] *
                                            W[:, :, :, k], axis=(1, 2, 3))
                output[:, y, x, k] = \
                    (activation(output[:, y, x, k] +
                                b[0, 0, 0, k]))
    return (output)
