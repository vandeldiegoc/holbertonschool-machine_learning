#!/usr/bin/env python3
"""module"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """ that randomly changes the brightness of an image:"""
    new_image = tf.image.adjust_brightness(image, max_delta)
    return new_image
