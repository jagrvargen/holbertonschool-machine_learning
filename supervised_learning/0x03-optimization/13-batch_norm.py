#!/usr/bin/env python3
"""
Contains the function def batch_norm(Z, gamma, beta, epsilon)
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes the unactivated output of a neural network.
    Z: numpy.ndarray (m, n) - Outputs to be normalized.
    gamma: numpy.ndarray (1, n) - Contains the scales used for batch norm.
    beta: numpy.ndarray (1, n) - Contains the offsets used for batch norm.
    epsilon: float - Used to prevent division by zero.

    Returns: The normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z- mean) / (np.sqrt(variance) + epsilon)
    
    return gamma * Z_norm + beta
