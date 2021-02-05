#!/usr/bin/env python3
"""module """
import tensorflow as tf


def lenet5(x, y):
    """ LeNet-5 architecture using tensorflow """
    initializer = \
        tf.contrib.layers.variance_scaling_initializer()

    layer0 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              name='layer')(x)

    layer1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2))(layer0)

    layer2 = tf.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              padding='valid',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              name='layer')(layer1)

    layer3 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2))(layer2)

    layer3 = tf.layers.Flatten()(layer3)

    layer4 = tf.layers.Dense(units=120,
                             activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             name='layer')(layer3)

    layer5 = tf.layers.Dense(units=84,
                             activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             name='layer')(layer4)

    layer6 = tf.layers.Dense(units=10,
                             kernel_initializer=initializer,
                             name='layer')(layer5)

    loss = tf.losses.softmax_cross_entropy(y, layer6)

    y_pred = tf.nn.softmax(layer6)

    train_op = tf.train.AdamOptimizer(name='Adam').minimize(loss)

    y_pred_t = tf.argmax(y_pred, 1)
    y_t = tf.argmax(y, 1)
    equal = tf.equal(y_pred_t, y_t)
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    return y_pred, train_op, loss, accuracy
