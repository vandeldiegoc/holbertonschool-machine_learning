#!/usr/bin/env python3
""" module """

import numpy as np
import tensorflow as tf


class NST:
    """ new class """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ funtion """
        tf.enable_eager_execution()
        error1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        error2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(error1)
        if len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(error1)
        if not isinstance(content_image, np.ndarray):
            raise TypeError(error2)
        if len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(error2)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.executing_eagerly()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """ Returns: the scaled image """
        error1 = "image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(image, np.ndarray):
            raise TypeError(error1)
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(error1)
        max_dim = 512
        long_dim = max(image.shape[:-1])
        scale = max_dim / long_dim
        h, w = image.shape[:-1]
        nw_shape_h = int(scale * h)
        nw_shape_w = int(scale * w)
        nw_shape = (nw_shape_h, nw_shape_w)
        image = image[tf.newaxis, :]
        image = tf.image.resize_bicubic(image, nw_shape)
        image = image / 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        return image
