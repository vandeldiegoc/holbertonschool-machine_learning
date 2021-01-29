#!/usr/bin/env python3
""" module """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ that performs a valid convolution on grayscale images: """
    m, h, w = images.shape
    hh, wh = kernel.shape
    ouput_h = h - hh + 1
    ouput_w = w - wh + 1

    ouput = np.zeros((m, ouput_h, ouput_w))

    for y in range(ouput_h):
        for x in range(ouput_w):
            ouput[:, y, x] = (kernel *
                              images[:, y: y + hh, x: x + wh])\
                                .sum(axis=(1, 2))
    return(ouput)
