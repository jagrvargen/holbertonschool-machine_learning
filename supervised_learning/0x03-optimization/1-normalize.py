#!/usr/bin/env python3
"""
Contains the function def normalize(X, m, v, epsilon)
"""
import numpy as np


def normalize(X, m, v, epsilon):
    """
    Normalizes a matrix
    X: numpy.ndarray (d, nx) - Where d is the number of data points and
    nx is the number of features/
    m: numpy.ndarray (nx,) - Contains the mean of all features of X
    v: numpy.ndarray (nx,) - Contains the variance of all features of X
    epsilon: float - Corrective value to prevent division by 0

    returns: X - The normalized matrix X
    """
    return (X - m) / np.sqrt(v + epsilon)
