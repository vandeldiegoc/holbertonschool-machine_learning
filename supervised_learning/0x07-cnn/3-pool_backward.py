#!/usr/bin/env python3
""" module """

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """that performs back propagation over
       a pooling layer of a neural network:"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)

    for z in range(m):
        for y in range(h_new):
            for x in range(w_new):
                for k in range(c_new):
                    images = A_prev[z, y * sh:(kh+y*sh), x * sw:(kw+x*sw), k]
                    tmp_dA = dA[z, y, x, k]
                    if mode == 'max':
                        z_mask = np.zeros(kernel_shape)
                        v_max = np.max(images)
                        np.place(z_mask, images == v_max, 1)
                        dA_prev[z, y * sh:(kh + y * sh),
                                x * sw:(kw+x*sw), k] += z_mask * tmp_dA

                    elif mode == 'avg':
                        avg = tmp_dA / kh / kw
                        dA_prev[z, y * sh:(kh + y * sh),
                               x * sw:(kw+x*sw),
                               k] += np.ones(kernel_shape) * avg
    return dA_prev
