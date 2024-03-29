#!/usr/bin/env python3
""" funtion that creates the forward
 propagation graph for the neural network"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Returns: the prediction of the network in tensor form"""
    layers = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        layers = create_layer(layers,
                              layer_sizes[i],
                              activations[i])
    return layers
