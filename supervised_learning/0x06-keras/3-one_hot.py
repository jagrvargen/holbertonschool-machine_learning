#!/usr/bin/env python3
"""
Contains the function def one_hot(labels, classes=None)
"""
import tensorflow as tf
import tensorflow.keras as keras


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.
    labels: np.ndarray - Contains the numerical labels for the data.
    classes: int - The number of classes in the data.

    Returns: The one-hot matrix.
    """
    return keras.utils.to_categorical(labels)
