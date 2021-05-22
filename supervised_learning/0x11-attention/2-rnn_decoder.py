#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       kernel_initializer='glorot_uniform', 
                                       return_sequences=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        x = self.embedding(x)
        g, state = self.gru(s_prev, hidden_states[2])  


        
