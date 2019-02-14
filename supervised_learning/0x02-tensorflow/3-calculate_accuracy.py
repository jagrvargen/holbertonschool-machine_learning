#!/usr/bin/env python3
"""
Contains the function def calculate_accuracy(y, y_pred)
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of the prediction made by the neural network.
    y: tf.placeholder - Contains the labels for the input data.
    y_pred: tf.Tensor - Contains the network's predictions.

    returns: tf.Tensor - Contains the decimal accuracy of the prediction.
    """
    true_vals = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(true_vals, tf.float32))

    return accuracy
