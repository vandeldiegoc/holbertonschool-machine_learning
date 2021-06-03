#!/usr/bin/env python3
""" module """
import tensorflow_datasets as tfds
import numpy
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    seq_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = seq_mask[:, tf.newaxis, tf.newaxis, :]

    seq2_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = seq2_mask[:, tf.newaxis, tf.newaxis, :]

    size = target.shape[1]

    look_ahead_mask = \
        1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decode_t_mask = target_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(decode_t_mask, look_ahead_mask)
    return encoder_mask, combined_mask, decoder_mask
