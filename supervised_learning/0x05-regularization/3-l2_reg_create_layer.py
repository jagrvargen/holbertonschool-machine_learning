#!/usr/bin/env python3
"""
Contains the function def l2_reg_create_layer(prev, n, activation, lambtha)
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer that includes L2 regularization.
    prev: tf.Tensor - Contains the output of the previous layer.
    n: int - The number of nodes the new layer will contain.
    activation: tf.nn.<layer> - The activation function for the new layer.
    lambtha: - float - The L2 regularization parameter.

    Returns: The output of the new layer.
    """
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), kernel_regularizer=tf.contrib.layers.l2_regularizer(lambtha),  name="layer")

    return layer(prev)



