#!/usr/bin/env python3
"""module """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ the network’s prediction, accuracy, and loss, respectively """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy = tf.get_collection("accuracy")[0]
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss = tf.get_collection("loss")[0]
        cost = sess.run(loss, feed_dict={x: X, y: Y})

        return prediction, accuracy, cost
