#!/usr/bin/env python3
"""
Contains the function def l2_reg_cost(cost)
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization.
    cost: tf.Tensor - Contains the cost of the network without L2
    regularization.

    Returns: A tensor containing the cost of the network accounting for L2
    regularization.
    """
    return tf.losses.get_regularization_losses()(cost)
