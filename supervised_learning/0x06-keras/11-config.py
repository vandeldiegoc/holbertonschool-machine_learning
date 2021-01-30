#!/usr/bin/env python3
""" module """

import tensorflow.keras as k


def save_config(network, filename):
    """ saves a model’s configuration in JSON format """
    json = network.to_json()
    with open(filename, 'w') as f:
        f.write(json)
    return(None)


def load_config(filename):
    """ loads a model with a specific configuration """
    with open(filename, "r") as f:
        json = f.read()
    network = k.models.model_from_json(json)
    return network
