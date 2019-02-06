#!/usr/bin/env python3
"""
Contains def one_hot_decode(one_hot)
"""
import numpy as np


def one_hot_decode(one_hot):
    """ Converts a one-hot matrix into a vector of labels """
    labels = np.zeros((one_hot.shape[0],), dtype=int)

    for i in range(len(one_hot)):
        col = one_hot[:,i]
        for j, n in enumerate(col):
            if n == 1:
                labels[i] += j

    return labels
