#!/usr/bin/env python3
"""
Contains the function def create_confusion_matrix(labels, logits)
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    labels: numpy.ndarray (m, classes) - Contains the correct labels for each
    data point in one-hot format.
    logits: numpy.ndarray (m, classes) - Contains the predicted labels for each
    data point in one-hot format.

    Returns: A confusion matrix (classes, classes) with row indices representing
    correct labels and column indices representing predicted labels.
    """
    confusion_matrix = np.zeros((labels.shape[1], logits.shape[1]))

    for i in range(len(labels)):
        print(np.argmax(labels[i]))
        confusion_matrix[i][np.argmax(labels[i])] += 1
            
    for j in range(len(logits)):
        confusion_matrix[np.argmax(logits[i])][i] += 1

    return confusion_matrix
