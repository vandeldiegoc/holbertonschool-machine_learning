#!/usr/bin/env python3
""" module """

import tensorflow.keras as k


def predict(network, data, verbose=False):
    """ makes a prediction using a neural network: """
    return network.predict(data, verbose=verbose)
