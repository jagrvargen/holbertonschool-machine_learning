#!/usr/bin/env python3
"""
Contains the function def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L)
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights of a neural network with L2 regularization using
    gradient descent.
    Y: numpy.ndarray (classes, m) - Contains the correct labels for the data.
    weights: dict - Contains the weights and biases of the network.
    cache: dict - Contains the outputs of each layer of the network.
    alpha: float - The learning rate.
    lambtha: float - The L2 regularization parameter.
    L: int - The number of layers in the network.
    """
    m = Y.shape[1]
    dZ = cache["A{}".format(L)] - Y

    for l in range(L, 0, -1):
        dW = dZ @ cache["A{}".format(l-1)].T / m + lambtha / m * weights["W{}".format(l)]

        db = np.sum(dZ, axis=1, keepdims=True) / m
        dZ = weights["W{}".format(l)].T @ dZ * (cache["A{}".format(l-1)] * (1 - cache["A{}".format(l-1)]))

        weights["W{}".format(l)] -= alpha * dW
        weights["b{}".format(l)] -= alpha * db
