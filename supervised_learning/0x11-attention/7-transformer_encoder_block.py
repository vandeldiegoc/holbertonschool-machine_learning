#!/usr/bin/env python3
""" class """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention

class EncoderBlock(tf.keras.layers.Layer):
    """to create an encoder block for a transformer:"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        '''costructor '''
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """initialize decoder """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.dense_hidden(out1)

        out2 = self.dense_output(ffn_output)
        ffn_output = self.dropout2(out2, training=training)
        output = self.layernorm2(out1 + ffn_output)
        return output
