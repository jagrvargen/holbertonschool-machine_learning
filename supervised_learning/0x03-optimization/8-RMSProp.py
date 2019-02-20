#!/usr/bin/env python3
"""
Contains the function def create_RMSProp_op(loss, alpha, beta2, epsilon)
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network using the RMSProp
    optimization algorithm.
    loss: tf.Tensor - The loss of the network.
    alpha: float - The learning rate.
    beta2: float - The RMSProp weight.
    epsilon: float - Used to prevent division by zero.

    Returns: The RMSProp optimization operation.
    """
    return tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon).minimize(loss)
