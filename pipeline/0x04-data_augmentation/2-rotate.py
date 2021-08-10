#!/usr/bin/env python3
"""module"""
import tensorflow as tf


def rotate_image(image):
    """that rotates an image by 90 degrees counter-clockwise:"""
    new_image = tf.image.rot90(image, 1)
    return new_image
