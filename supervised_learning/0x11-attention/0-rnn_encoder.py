#!/usr/bin/env python3
""" class """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ class encode """
    def __init__(self, vocab, embedding, units, batch):
        """ constuctor """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
             units=self.units,
             recurrent_initializer='glorot_uniform',
             return_sequences=True,
             return_state=True)

    def initialize_hidden_state(self):
        """Returns: a tensor of shape
            (batch, units)containing the
            initialized hidden states"""
        return tf.zeros([self.batch, self.units])

    def call(self, x, initial):
        """crate and return encode output"""
        inp = self.embedding(x)
        ouput, status = self.gru(inp, initial_state=initial)
        return ouput, status
