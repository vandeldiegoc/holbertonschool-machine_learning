#!/usr/bin/env python3
"""module"""
import tensorflow as tf


def flip_image(image):
    """that flips an image horizontally:"""
    new_image = tf.image.flip_left_right(image)
    return(new_image)
