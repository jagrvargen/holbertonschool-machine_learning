#!/usr/bin/env python3
"""
Contains the function def create_Adam_op(loss, alpha, beta1, beta2, epsilon)
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network using the Adam
    optimization algorithm.
    loss: tf.Tensor - The loss of the network.
    alpha: float - The learning rate.
    beta1: float - The weight used for the first moment.
    beta2: float - The weight used for the second moment.
    epsilon: float - Used to prevent division by zero

    Returns: The Adam optimization operation
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
