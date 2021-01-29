#!/usr/bin/env python3
""" module """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """that performs a convolution on grayscale images"""
    m, h, w = images.shape
    hh, wh = kernel.shape
    sw, sh = stride[1], stride[0]
    if padding == 'valid':
        ph = 0
        pw = 0

    if padding == 'same':
        ph = int(((h - 1) * sh + hh - h) / 2) + 1
        pw = int(((w - 1) * sw + wh - w) / 2) + 1

    if type(padding) == tuple:
        ph = padding[0]
        pw = padding[1]
    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)],
                    mode='constant', constant_values=0)

    ouput_h = int(float(h - hh + (2 * ph)) / float(sh)) + 1
    ouput_w = int(float(w - wh + (2 * pw)) / float(sw)) + 1

    ouput = np.zeros((m, ouput_h, ouput_w))

    for y in range(ouput_h):
        for x in range(ouput_w):
            ouput[:, y, x] = (kernel * images[:,
                              y*(stride[0]): y*(stride[0]) + hh,
                              x*(stride[1]): x*(stride[1]) + wh])\
                                        .sum(axis=(1, 2))
    return(ouput)
