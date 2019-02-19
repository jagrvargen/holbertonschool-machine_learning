#!/usr/bin/env python3
"""
Contains the function def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s)
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.
    alpha: float - The learning rate.
    beta2: float - The RMSProp weight.
    epsilon: float - Used to avoid division by zero.
    var: numpy.ndarray - Contains the variable to be updated.
    grad: numpy.ndarray - Contains the gradient of var.
    s: numpy.ndarray - The previous second moment of var.

    Returns: The updated variable and the new moment, respectively.
    """
    Sd = beta2 * s + (1 - beta2) * (grad ** 2)
    var_updated = var - alpha * (grad / (np.sqrt(Sd) + epsilon))

    return var_updated, Sd
