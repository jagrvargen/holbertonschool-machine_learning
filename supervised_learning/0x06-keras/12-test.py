#!/usr/bin/env python3
"""
Contains the function def test_model(network, data, labels, verbose=True)
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a Keras neural network.
    network: keras.Model - The Keras model to be tested.
    data: numpy.ndarray - The input data to test the model with.
    labels: numpy.ndarray - The correct one-hot labels of the data.
    verbose: bool - Determines whether or not to print verbose output.

    Returns: The loss and accuracy of the model with the testing data.
    """
    return network.evaluate(data, labels, verbose=verbose)
