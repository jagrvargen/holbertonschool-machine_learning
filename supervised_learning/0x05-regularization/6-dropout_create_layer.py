#!/usr/bin/env python3
"""
Contains the function def dropout_create_layer(prev, n, activation, keep_prob)
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout.
    prev: tf.Tensor - Contains the output of the previous layer.
    n: int - The number of nodes the new layer should contain.
    activation: tf.nn.<activation> - The activation function to be used.
    keep_prob: float - The probability that a node will be dropped.

    Returns: The output of the new layer.
    """
    layer = tf.layers.Dense(n, activation, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), name="layer")
    A = layer(prev)
    dropout = tf.layers.Dropout((1 - keep_prob),  name="dropout")

    return dropout(A)
