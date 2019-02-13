#!/usr/bin/env python3
"""
Contains the function def create_train_op(loss, alpha)
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the neural network.
    loss: tf.Tensor - Contains the loss of the network's prediction.
    alpha: float - The learning rate of the network.

    returns: tf.Operation - Trains the network using gradient descent.
    """
    op = tf.train.GradientDescentOptimizer(alpha)

    return op.minimize(loss)
