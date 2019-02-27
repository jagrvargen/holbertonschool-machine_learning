#!/usr/bin/env python3
"""
Contains the function def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L)
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with dropout regularization using
    gradient descent.
    Y: numpy.ndarray (classes, m) - Contains the correct labels for the data.
    weights: dict - Contains the weights and biases for the network.
    cache: dict - Contains the outputs and dropout masks for the network.
    alpha: float - The learning rate.
    keep_prob: float - The probability that a node will be kept.
    L: int - The number of layers in a network.

    Returns: Nothing. The weights are updated in place.
    """
    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y
    dW = dZ @ cache["A{}".format(L-1)].T / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    weights["W{}".format(L)] -= alpha * dW
    weights["b{}".format(L)] -= alpha * db

    for l in range(L-1, 0, -1):
        dA = weights["W{}".format(l+1)].T @ dZ
        dA *= cache["D{}".format(l)] / keep_prob
        A = cache["A{}".format(l)]
        dZ = dA * (1 - (A ** 2))
        dW = dZ @ cache["A{}".format(l-1)].T / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W{}".format(l)] -= alpha * dW
        weights["b{}".format(l)] -= alpha * db
