#!/usr/bin/env python3
"""
Contains the function def evaluate(X, Y, save_path)
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of the neural network.
    X: numpy.ndarray - Contains the input data to evaluate.
    Y: numpy.ndarray - Contains the one-hot labels for X.
    save_path: string - Path to load the model from.

    returns: The network's prediction, accuracy, and loss
    """
    with tf.Session() as sess:
        
