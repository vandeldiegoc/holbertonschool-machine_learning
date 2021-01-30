#!/usr/bin/env python3
"""save_model, load_model"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model"""
    network.save(filename)
    return(None)


def load_model(filename):
    """Loads an entire model"""
    loaded = K.models.load_model(filename)
    return(loaded)
