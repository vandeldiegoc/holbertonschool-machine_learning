import numpy as np
import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1, activation='relu')
    def call(self, s_prev, hidden_states):
       h = self.W(hidden_states)
       s_newdims = tf.expand_dims(s_prev, axis=1)
       s = self.U(s_newdims)

       e = self.V(h+s)
       a = tf.keras.activations.softmax(e)
       print(a.shape)
       print(hidden_states.shape)
       c = tf.reduce_sum(a * hidden_states, 1)
       return c, e

    




        

