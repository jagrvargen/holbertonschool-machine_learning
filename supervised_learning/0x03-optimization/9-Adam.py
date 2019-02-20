#!/usr/bin/env python3
"""
Contains the function def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t)
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.
    alpha: float - The learning rate.
    beta1: float - The weight used for the first moment.
    beta2: float - The weight used for the second moment.
    epsilon: float - Used to prevent division by zero.
    var: numpy.ndarray - Contains the variable to be updated.
    grad: numpy.ndarray - Contains the gradient of var.
    v: float - The previous first moment of var.
    s: float - The previous second moment of var.
    t: int - The time step used for bias correction.

    Returns: The updated variable, the new first moment, and the second new
    moment, respectively.
    """
    Vd = beta1 * v + (1 - beta1) * grad
    Sd = beta2 * s + (1 - beta2) * (grad ** 2)

    Vd_corrected = Vd / (1 - beta1 ** t)
    Sd_corrected = Sd / (1 - beta2 ** t)

    var_updated = var - alpha * (Vd / (np.sqrt(Sd) + epsilon))

    return var_updated, Vd, Sd
