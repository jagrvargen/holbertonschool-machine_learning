#!/usr/bin/env python3
"""
Contains the function def l2_reg_cost(cost, lambtha, weights, L, m)
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    cost: float - The cost of the network without L2 regularization.
    lambtha: float - The regularization parameter.
    weights: dict - A dictionary containing numpy.ndarrays(s) which hold the
    weights and biases of the neural network.
    L: int - The number of layers in the neural network.
    m: int - The number of data points used.

    Returns: The cost of the network
    """
    total = 0
    for matrix in weights.values():
        total += np.linalg.norm(matrix)

    return cost + lambtha / (2 * m) * total
