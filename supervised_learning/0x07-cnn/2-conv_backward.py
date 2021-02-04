#!/usr/bin/env python3
""" module """

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """that performs forward propagation
       over a pooling layer of a neural network"""
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    pw, ph = 0, 0

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        ph = int(np.ceil(((h_prev-1)*sh+kh-h_prev)/2))
        pw = int(np.ceil(((w_prev-1)*sw+kw-w_prev)/2))

    A_prev = np.pad(A_prev,
                    pad_width=((0, 0),
                               (ph, ph),
                               (pw, pw),
                               (0, 0)),
                    mode='constant', constant_values=0)

    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)

    for z in range(m):
        for y in range(h_new):
            for x in range(w_new):
                for k in range(c_new):
                    tmp_w = W[:, :, :, k]
                    tmp_dz = dZ[z, y, x, k]
                    dA[z, y*sh: y*sh+kh, x*sw: x*sw+kw, :] += tmp_dz * tmp_w
                    aux_A_prev = A_prev[z, y*sh: y*sh+kh, x*sw: x*sw+kw, :]
                    dW[:, :, :, k] += aux_A_prev * tmp_dz

    dA = dA[:, ph:dA.shape[1] - ph, pw:dA.shape[2] - pw, :]
    return dA, dW, db
