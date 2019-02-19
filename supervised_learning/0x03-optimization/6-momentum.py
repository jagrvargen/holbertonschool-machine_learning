#!/usr/bin/env python3
"""
Contains the function def create_momentum_op(loss, alpha, beta1)
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates a training operation for a neural network in TensorFlow using
    the gradient descent with momentum optimization algorithm.
    loss: tf.Operation - The loss of the network.
    alpha: float - The learning rate.
    beta1: float - The momentum weight.

    Returns: The momentum optimization operation
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
