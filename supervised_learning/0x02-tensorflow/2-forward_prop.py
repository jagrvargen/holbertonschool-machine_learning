#!/usr/bin/env python3
"""
Contains the function def forward_prop(x, nx, layer_sizes=[], activations=[])
"""
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, nx, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network
    x: tf.placeholder - Holds the input data
    nx: int - The number of input columns
    layer_sizes: list - Contains the number of nodes in each layer of the
    network
    activations: list - Contains the activation functions for each layer of
    the network

    returns: tensor - Prediction of the network
    """
    with tf.name_scope("layer"):
        layer = create_layer(x, x.shape[1], layer_sizes[1], activations[0])

        for i in range(1, len(layer_sizes)):
            layer = create_layer(layer[:], layer.shape[1], layer_sizes[i], activations[i])

        return layer
