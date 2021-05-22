#!/usr/bin/env python3
""" class """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ to calculate the attention"""
    def __init__(self, units):
        """ constructor """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1, activation='relu')

    def call(self, s_prev, hidden_states):
        """ calculate self-atention """
        s_newdims = self.W(tf.expand_dims(s_prev, axis=1))
        h = self.U(hidden_states)
        e = self.V(tf.tanh(h+s_newdims))
        a = tf.keras.activations.softmax(e, axis=1)
        c = tf.reduce_sum(a * hidden_states, 1)
        return c, a