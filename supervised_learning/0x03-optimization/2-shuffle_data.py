#!/usr/bin/env python3
"""
Contains the function def shuffle_data(X, Y)
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the datapoints of two matrices in parallel
    X: numpy.ndarray (m, nx) - Where m is the number of data points and
    nx is the number of features.
    Y: numpy.ndarray (m, ny) - Where m is the number of data points and
    ny is the number of features.

    Returns: Shuffled X and Y
    """
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    return X_shuffled, Y_shuffled
