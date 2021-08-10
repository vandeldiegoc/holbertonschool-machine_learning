#!/usr/bin/env python3
"""module"""
import tensorflow as tf


def change_hue(image, delta):
    """ that changes the hue of an image:"""
    new_image = tf.image.adjust_hue(image, delta)
    return new_image
