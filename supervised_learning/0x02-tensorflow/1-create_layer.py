#!/usr/bin/env python3
"""
Contains the funtion def create_layer(prev, n_prev, n, activation)
"""
import tensorflow as tf


def create_layer(prev, n_prev, n, activation):
    """
    Creates a layer for a neural network
    prev: tensor - Output from previous layer
    n_prev: int - Number of nodes in the previous layer
    n: int - Number of nodes to instantiate in new layer
    activation: tf.nn.{activation} - Activation function for new layer
    """
    with tf.name_scope("layer"):
        layer = tf.layers.Dense(n, activation=activation, use_bias=True, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), name="layer")

    return layer(prev)
