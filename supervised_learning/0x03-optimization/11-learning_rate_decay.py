#!/usr/bin/env python3
"""
Contains the function def learning_rate_decay(alpha, decay_rate, global_step, decay_step)
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay.
    alpha: float - The learning rate.
    decay_rate: float - The weight used to determine the rate at which alpha
    will decay.
    global_step: int - The number of passes of gradient descent that have
    elapsed.
    decay_step: int - The number of passes of gradient descent that should
    occur before alpha is decayed further.

    Returns: The updated value for alpha.
    """
    decay_rate /= global_step // decay_step + 1
    alpha *= decay_rate

    return alpha
