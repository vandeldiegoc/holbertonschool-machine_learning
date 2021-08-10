#!/usr/bin/env python3
"""module"""
import tensorflow as tf


def crop_image(image, size):
    """ that performs a random crop of an image:j"""
    new_image = tf.image.random_crop(image, size)
    return new_image
