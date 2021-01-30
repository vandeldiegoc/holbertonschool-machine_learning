#!/usr/bin/env python3
"""save_model, load_model"""
import tensorflow.keras as k


def save_model(network, filename):
    """Saves an entire model"""
    network.save(filename)
    return(None)


def load_model(filename):
    """Loads an entire model"""
    loaded = k.models.load_model(filename)
    return(loaded)
