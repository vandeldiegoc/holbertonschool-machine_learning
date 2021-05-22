#!/usr/bin/env python3
""" class """
import numpy as np
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ class encode """
    def __init__(self, vocab, embedding, units, batch):
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units=self.units,
                                       kernel_initializer='glorot_uniform',
                                       return_state=True)

    def initialize_hidden_state(self):
        initializer = tf.keras.initializers.Zeros()
        layer = initializer(shape=(self.batch, self.units))
        return layer

    def call(self, x, initial):
        inp = self.embedding(x)
        ouput, status = self.gru(inp, initial_state=initial)
        return ouput, status
