#!/usr/bin/env python3
""" This file contains the np_cat function. """
import numpy as np


def np_concatenate(mat1, mat2, axis=0):
    """ Concatenates two tensors along a specific axis. """
    return np.concatenate((mat1, mat2), axis=axis)
