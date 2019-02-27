#!/usr/bin/env python3
"""
Contains the function def early_stopping(cost, prev_cost, tolerance, patience, count)
"""


def early_stopping(cost, prev_cost, tolerance, patience, count):
    """
    Determines if gradient descent should be stopped early.
    cost: float - The current cost of the network.
    prev_cost: float - The previous recorded cost of the network.
    tolerance: float - The threshold used to detrmine early stopping.
    patience: int - The maximum number of epochs before employing early stopping.
    count: int - The current patience count.

    Returns: A tuple containing a boolean indicating whether the network should be
    stopped early followed by the updated patience count.
    """
    if count >= patience - 1 and cost - prev_cost >= tolerance:
        return True, count + 1
    elif count < patience - 1:
        return False, count + 1
    elif cost <= prev_cost:
        return False, 0
        
