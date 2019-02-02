#!/usr/bin/env python3
"""
Contains def one_hot_encode(Y, classes)
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Creates a (classes, m) one-hot encoding matrix from an m-dimensional matrix
    """
    one_hot_matrix = np.zeros((classes, Y.shape[0]))

    for i, n in enumerate(Y):
        one_hot_matrix[n][i] += 1

    return one_hot_matrix
