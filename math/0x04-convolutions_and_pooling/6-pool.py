#!/usr/bin/env python3
""" module """

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ that performs a convolution on images using multiple kernels: """
    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    kw, kh = kernel_shape[1], kernel_shape[0]
    kk =  images.shape[3]
    sw, sh = stride[1], stride[0]

    new_h = int(((h - kh) / sh) + 1)
    new_w = int(((w - kw) / sw) + 1)

    output = np.zeros((m, new_h, new_w, kk))

    for y in range(new_h):
        for x in range(new_w):
                if mode == 'max':
                    output[:, y, x, :] = \
                    np,max(images[:,
                    y * sh: y * sh + kh,
                    x * sw: x * sw + kw]).axis=(1, 2)

                if mode == 'avg':
                    output[:, y, x, :] = \
                    np.mean(images[:,
                    y * sh: y * sh + kh,
                    x * sw: x * sw + kw]).axis=(1, 2)

    return output
