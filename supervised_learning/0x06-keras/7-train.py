#!/usr/bin/env python3
"""
Contains the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True)
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True):
    """
    Trains a model using mini-batch gradient descent.
    network: keras.Model - The model to be trained.
    data: numpy.ndarray - Contains the input data for the network.
    labels: numpy.ndarray - Contains the labels for the data.
    batch_size: int - The size of the batches used in mini-batch gradient descent.
    epochs: int - The number of passes the model makes through the network.
    validation_data: tuple (numpy.ndarray, numpy.ndarray) - Contains the
    validation data and labels used to validate the model.
    early_stopping: bool - Indicates whether early stopping should be used
    (if validation_data exists).
    patience: int - The patience used for early stopping.
    learning_rate_decay: bool - Indicates whether or not learning rate decay
    will take place.
    alpha: float - The initial learning rate.
    decay_rate: float The rate at which learning rate decay will take place.
    verbose: bool - Used to determine if output should be printed during training.

    Returns: Nothing.
    """
    callbacks = []
    if validation_data:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(patience=patience, verbose=verbose))
        if learning_rate_decay:
            def schedule(epoch):
                return alpha / (1 + decay_rate * epoch)
            callbacks.append(K.callbacks.LearningRateScheduler(schedule=schedule, verbose=verbose))

    network.fit(data, labels, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=validation_data, verbose=verbose)
