#!/usr/bin/env python3
"""
Contains the function def specificity(confusion)
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.
    confusion: numpy.ndarray (classes, classes) - Where rows represent the
    correct labels and columns represent the predicted labels.

    Returns: A vector containing the specificity of each class.
    """
    specificity_measures = np.zeros((confusion.shape[0],))

    for i, row in enumerate(confusion):
        specificity_measures[i] = (np.sum(confusion[:,i]) - row[i]) / ((np.sum(confusion[:,i]) - row[i]) + (np.sum(row) - row[i]))
    
    return specificity_measures
