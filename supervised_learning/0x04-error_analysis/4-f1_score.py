#!/usr/bin/env python3
"""
Contains the function def f1_score(confusion)
"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix.
    confusion: numpy.ndarray (classes, classes) - Where the rows represent
    the correct labels and the columns represent the predicted labels.
    """
    f1 = 2 * (precision(confusion) * sensitivity(confusion)) / (precision(confusion) + (sensitivity(confusion)))

    return f1
