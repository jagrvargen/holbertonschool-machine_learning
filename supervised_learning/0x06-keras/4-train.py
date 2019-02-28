#!/usr/bin/env python3
"""
Contains the function def train_model(network, data, labels, batch_size, epochs, verbose=True)
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True):
    """
    Trains a model using gradient descent.
    network: K.Model - The model to train.
    data: numpy.ndarray (m, nx) - Contains the input data.
    labels: numpy.ndarray (m, classes) - Contains the correct labels for the data.
    batch_size: int - The size of the batch used for mini-batch gradient descent.
    epochs: int - The number of passes through data for gradient descent.
    verbose: bool - Determines if output should be printed during training.

    Returns: Nothing
    """
    network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose)
