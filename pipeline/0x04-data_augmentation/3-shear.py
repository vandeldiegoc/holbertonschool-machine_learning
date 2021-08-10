#!/usr/bin/env python3
"""module"""
import tensorflow as tf


def shear_image(image, intensity):
    """  that randomly shears an image: """
    array = tf.keras.preprocessing.image.random_shear(image, intensity)
    new_image = tf.keras.preprocessing.image.array_to_img(array)
    return new_image
