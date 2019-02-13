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
        W = tf.get_variable("W", shape=(n_prev, n), initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), dtype=tf.float32)
        b = tf.get_variable("b", shape=(n,), dtype=tf.float32)
        layer = activation((tf.matmul(prev, W) + b))

    return layer
