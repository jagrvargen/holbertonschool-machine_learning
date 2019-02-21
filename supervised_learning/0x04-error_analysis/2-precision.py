#!/usr/bin/env python3
"""
Contains the function def precision(confusion)
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.
    confusion: numpy.ndarray (classes, classes) - Where rows represent
    the correct labels and columns represent the predicted labels.

    Returns: A vector containing the precision of each class.
    """
    precision_measures = np.zeros((confusion.shape[0],))

    for i, row in enumerate(confusion):
        precision_measures[i] = row[i] / np.sum(confusion[:,i])

    return precision_measures
