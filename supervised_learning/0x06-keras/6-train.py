#!/usr/bin/env python3
"""module"""

import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """  that trains a model using mini-batch gradient descent: """
    callbacks = []
    if validation_data:
        es = k.callbacks.EarlyStopping(monitor='val_loss',
                                       patience=patience)
        callbacks.append(es)

    output = network.fit(data,
                         labels,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=validation_data,
                         verbose=verbose,
                         shuffle=shuffle,
                         callbacks=callbacks)
    return output
