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
             recurrent_initializer='glorot_uniform',
             return_sequences=True,
             return_state=True
             )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """decode"""
        encode = SelfAttention(self.units)
        c, prev_state = encode(s_prev, hidden_states)
        x = self.embedding(x)
        inputs = tf.concat([tf.expand_dims(c, 1), x], axis=-1)
        decode_output, s = self.gru(inputs)
        output = tf.reshape(decode_output, (-1, decode_output.shape[2]))
        y = self.F(output)
        return y, s
