#!/usr/bin/env python3
"""
Contains the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True)
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True):
    """
    Trains a model using mini-batch gradient descent.
    network: K.Model - The model to be trained.
    data: numpy.ndarray - Contains the input data for the network.
    labels: numpy.ndarray - Contains the labels for the data.
    batch_size: int - The size of the batches used in mini-batch gradient descent.
    epochs: int - The number of passes the model makes through the network.
    validation_data: tuple (numpy.ndarray, numpy.ndarray) - Contains the
    validation data and labels used to validate the model.
    early_stopping: bool - Indicates whether early stopping should be used
    (if validation_data exists).
    patience: int - The patience used for early stopping.
    verbose: bool - Used to determine if output should be printed during training.

    Returns: Nothing.
    """
    if early_stopping:
        network.fit(data, labels, batch_size=batch_size, epochs=epochs, callbacks=[K.callbacks.EarlyStopping(patience=patience)], validation_data=validation_data, verbose=verbose)
    else:
        network.fit(data, labels, batch_size=batch_size, epochs=epochs, validation_data=validation_data, verbose=verbose)
