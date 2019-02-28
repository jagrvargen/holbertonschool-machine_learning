#!/usr/bin/env python3
"""
Contains the function def predict(network, data, verbose=False)
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a trained neural network.
    data: numpy.ndarray - The data with which to make the prediction.
    verbose: bool - Determines whether verbose output should be printed.

    Returns: The prediction for the data.
    """
    return network.predict(data, verbose=verbose)
