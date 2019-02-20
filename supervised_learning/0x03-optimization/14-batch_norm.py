#!/usr/bin/env python3
"""
Contains the function def create_batch_norm_layer(prev, n, activation)
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.
    prev: tf.Tensor - The activated output of the previous layer.
    n: int - The number of nodes in the layer to be created.
    activation: tf.Operation - The activation function to be used on the output
    layer

    Returns: A tensor of the activated output for the layer.
    """
    layer = tf.layers.Dense(n, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), name="dense")
    z = layer(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma", trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name="beta", trainable=True)
    mean, variance = tf.nn.moments(z, axes=0)
    out = tf.nn.batch_normalization(z, mean, variance, beta, gamma, 1e-8)
    
    return activation(out)
