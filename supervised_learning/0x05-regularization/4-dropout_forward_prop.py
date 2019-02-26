#!/usr/bin/env python3
"""
Contains the function def dropout_forward_prop(X, weights, L, keep_prob)
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using dropout.
    X: numpy.ndarray (nx, m) - Contains the input data for the network.
    weights: dict - Contains the weights and biases of the neural network.
    L: int - The number of layers in the network.
    keep_prob: float - The probability that a node will be kept.

    Returns: A dictionary containing the outputs of each layer and the dropout
    mask used on each layer.
    """
    cache = {}
    cache["A0"] = X

    for l in range(1, L + 1):
        Z = weights["W{}".format(l)] @ cache["A{}".format(l-1)] + weights["b{}".format(l)]
        A = 1 / (1 + np.exp(-Z))

        if l < L:
            D = np.random.rand(A.shape[0], A.shape[1])
            cache["D{}".format(l)] = np.where(D < keep_prob, 1, 0)
            A *= cache["D{}".format(l)]
            A /= keep_prob

        cache["A{}".format(l)] = A

    return cache
