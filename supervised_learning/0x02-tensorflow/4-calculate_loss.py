#!/usr/bin/env python3
"""
Contains the function def calculate_loss(y, y_pred)
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of the neural network's
    prediction.
    y: tf.placeholder - Contains the labels for the input data.
    y_pred: tf.Tensor - Contains the network's predictions.

    returns: tf.Tensor - Contains the loss of the prediction.
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
