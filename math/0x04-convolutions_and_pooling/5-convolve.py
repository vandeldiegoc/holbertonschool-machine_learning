#!/usr/bin/env python3
""" module """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """that performs a convolution on grayscale images"""
    m, h, w, c = images.shape
    hh, wh, cc, kk = kernels.shape
    sw, sh = stride

    ph = 0
    pw = 0

    if padding == 'same':
        ph = int(((h - 1) * sh + hh - h) / 2) + 1
        pw = int(((w - 1) * sw + wh - w) / 2) + 1

    if type(padding) == tuple:
        ph = padding[0]
        pw = padding[1]
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)

    ouput_h = int(((h + 2 * ph - hh) / sh) + 1)
    ouput_w= int(((w + 2 * pw - wh) / sw) + 1)

    ouput = np.zeros((m, ouput_h, ouput_w, kk))

    for y in range(ouput_h):
        for x in range(ouput_w):
            for k in range(kk):
                ouput[:, y, x, k] = (kernels[:, :, :, k] * images[:,
                                     y*(stride[0]): y*(stride[0]) + hh,
                                     x*(stride[1]): x*(stride[1]) + wh])\
                                        .sum(axis=(1, 2, 3))
    return(ouput)
