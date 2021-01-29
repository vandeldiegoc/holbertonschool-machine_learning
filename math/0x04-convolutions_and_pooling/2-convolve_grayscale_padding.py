#!/usr/bin/env python3
""" module """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ that performs a convolution on grayscale images with custom paddin """
    m, h, w = images.shape
    hh, wh = kernel.shape
    ph = padding[0]
    pw = padding[1]
    ouput_h = h - hh + (2*ph) + 1
    ouput_w = w - wh + (2*pw) + 1

    ouput = np.zeros((m, ouput_h, ouput_w))

    images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)],
                    mode='constant', constant_values=0)

    for y in range(ouput_h):
        for x in range(ouput_w):
            ouput[:, y, x] = (kernel * images[:, y: y + hh, x: x + wh])\
                                        .sum(axis=(1, 2))
    return(ouput)
