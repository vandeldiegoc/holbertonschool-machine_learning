#!/usr/bin/env python3
""" module """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """that performs a same convolution on grayscale images"""
    m, h, w = images.shape
    hh, wh = kernel.shape
    ph = max(int((hh - 1) / 2), int(hh / 2))
    pw = max(int((wh - 1) / 2), int(wh / 2))

    images = np.pad(images,
                    pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant', constant_values=0)

    output = np.zeros((m, h, w))

    m = np.arange(0, m)

    for y in range(h):
        for x in range(w):
            output[m, y, x] = (kernel * images[m, y: y + hh, x: x + wh]).\
                    sum(axis=(1, 2))
    return(output)
