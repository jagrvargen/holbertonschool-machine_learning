#!/usr/bin/env python3
"""
Contains the function def normalization_constants(X)
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix.
    X: numpy.ndarray (m, nx) - Where m is the number of data points and
    nx is the number of features.

    returns: 
       mean: numpy.ndarray (1, nx)
       variance: numpy.ndarray (1, nx)
    """
    mean = np.sum(X, axis=0) / X.shape[0]
    variance = np.sum(np.square(X - mean), axis=0) / X.shape[0]

    return mean, variance
