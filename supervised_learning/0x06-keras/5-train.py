#!/usr/bin/env python3
"""module"""

import tensorflow.keras as k


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                verbose=True, shuffle=False):
    """  that trains a model using mini-batch gradient descent: """
    output = network.fit(data,
                         labels,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=validation_data,
                         verbose=verbose,
                         shuffle=shuffle)
    return output
