#!/usr/bin/env python3
"""
Contains the function def update_variables_momentum(alpha, beta1, var, grad, v)
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum algorithm.
    alpha: float - The learning rate.
    beta1: float - The momentum weight.
    var: numpy.ndarray - The variable to be updated.
    grad: numpy.ndarray - The gradient of var.
    v: The previous first moment of var.

    Returns: The updated variable and the new moment, respectively.
    """
    Vd = (beta1 * v) + (1 - beta1) * grad
    var_updated = var - alpha * Vd

    return var_updated, Vd
