#!/usr/bin/env python3
""" class """
import tensorflow as tf
import numpy as np


def sdp_attention(Q, K, V, mask=None):
    """that calculates the scaled dot product attention"""
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention = matmul_qk / dk
    if mask is not None:
        scaled_attention += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights
