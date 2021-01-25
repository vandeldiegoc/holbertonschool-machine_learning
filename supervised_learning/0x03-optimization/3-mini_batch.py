#!/usr/bin/env python3
""" module """

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ that trains a loaded neural network model
        using mini-batch gradient descent:
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        for i in range(epochs + 1):
            cost = sess.run(loss, feed_dict={x: X_train,
                                             y: Y_train})

            t_accuracy = sess.run(accuracy, feed_dict={x: X_train,
                                                       y: Y_train})

            v_cost = sess.run(loss, feed_dict={x: X_valid,
                                               y: Y_valid})

            v_accuracy = sess.run(
                accuracy,
                feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost))
            print("\tTraining Accuracy: {}".format(t_accuracy))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_accuracy))
            if i < epochs:

                x_s, y_s = shuffle_data(X_train, Y_train)
                m = x_s.shape[0]
                start = 0
                step = 1
                while m > 0:
                    if m - batch_size < 0:
                        end = x_s.shape[0]
                    else:
                        end = start + batch_size
                    sess.run(train_op, feed_dict={x: x_s[start:end],
                                                  y: y_s[start:end]})
                    if step % 100 == 0:
                        step_acc = sess.run(accuracy,
                                            feed_dict={x: x_s[start:end],
                                                       y: y_s[start:end]})
                        step_cost = sess.run(loss,
                                             feed_dict={x: x_s[start:end],
                                                        y: y_s[start:end]})
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_acc))
                    start += batch_size
                    step += 1
                    m -= batch_size
        save_p = saver.save(sess, save_path)
        return save_p
