#!/usr/bin/env python3
"""module"""

import tensorflow.keras as k


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """  that trains a model using mini-batch gradient descent: """
    def learning_rate(epoch):
        """ the decay should be performed using inverse time decay"""
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if save_best:
        best_save = k.callbacks.ModelCheckpoint(filepath,
                                                save_best_only=True,
                                                monitor='val_loss',
                                                mode='min')
        callbacks.append(best_save)

    if (validation_data and learning_rate_decay):
        es = k.callbacks.LearningRateScheduler(learning_rate, 1)
        callbacks.append(es)

    if validation_data and early_stopping:
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
