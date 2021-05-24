#!/usr/bin/env python3
"""module """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ class """
    def __init__(self, vocab, embedding, units, batch):
        """contrutor """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
             units=units,
             return_sequences=True,
             return_state=True,
             recurrent_initializer='glorot_uniform'
             )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """decode"""
        encode = SelfAttention(self.units)
        c, prev_state = encode(s_prev, hidden_states)
        x = self.embedding(x)
        c = tf.expand_dims(c, axis=1)
        inputs = tf.concat([c, x], axis=-1)
        decode_output, prev_hidden = self.gru(inputs)
        output = tf.reshape(decode_output, (-1, y.shape[2]))
        y = self.F(y)
        return y, prev_hidden
