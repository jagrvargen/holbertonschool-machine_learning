#!/usr/bin/env python3
"""
Contains the function def sensitivity(confusion)
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.
    confusion: numpy.ndarray - A confusion matrix where rows represent the
    correct labels and columns represent the predicted labels.

    Returns: A vector containing the sensitivity for each class.
    """
    sensitivity_measures = np.zeros((confusion.shape[0],))

    for i, row in enumerate(confusion):
        sensitivity_measures[i] = row[i] / np.sum(row)

    return sensitivity_measures
