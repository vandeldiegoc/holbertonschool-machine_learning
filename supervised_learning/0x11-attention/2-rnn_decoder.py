#!/usr/bin/env python3
"""module """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ class """
    def __init__(self, vocab, embedding, units, batch):
        """contrutor """
        super(RNNDecoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
             units=units,
             kernel_initializer='glorot_uniform',
             return_sequences=True,
             return_state=True,)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """decode"""
        encode = SelfAttention(self.units)
        x = self.embedding(x)
        c, prev_state = encode(s_prev, hidden_states)
        c = tf.expand_dims(c, axis=1)
        inputs = tf.concat([c, x], axis=-1)
        y = self.gru(inputs)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.F(y)
        return y, prev_stat
