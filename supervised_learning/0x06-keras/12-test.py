#!/usr/bin/env python3
"""module"""
import tensorflow.keras as k


def test_model(network, data, labels, verbose=True):
    """tests a neural network"""
    return network.evaluate(data, labels, verbose=verbose)
