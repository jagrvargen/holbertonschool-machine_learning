#!/usr/bin/env python3
"""
Contains the function def one_hot(labels, classes=None)
"""
import numpy as np


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.
    labels: np.ndarray - Contains the numerical labels for the data.
    classes: int - The number of classes in the data.

    Returns: The one-hot matrix.
    """
    one_hot = np.zeros((len(labels), np.max(labels) + 1))

    for i, label in enumerate(labels):
        one_hot[i][label] += 1

    return one_hot
