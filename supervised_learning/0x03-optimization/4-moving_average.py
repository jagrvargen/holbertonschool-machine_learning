#!/usr/bin/env python3
"""
Contains the function def moving_average(data, beta)
"""
import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.
    data: list - A list of numbers representing data.
    beta: float - The weight used for the moving average.

    Returns: A list of the weighted averages.
    """
    weighted_averages = []
    V = 0

    for i, val in enumerate(data):
        V = ((beta * V) + (1 - beta) * val)
        V_corrected = V / (1 - np.power(beta ,i + 1))
        weighted_averages.append(V_corrected)

    return weighted_averages
