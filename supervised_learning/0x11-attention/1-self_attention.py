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
        h = self.W(hidden_states)
        s_newdims = tf.expand_dims(s_prev, axis=1)
        s = self.U(s_newdims)
        e = self.V(h+s)
        a = tf.keras.activations.softmax(e)
        c = tf.reduce_sum(a * hidden_states, 1)
        return c, e





        

