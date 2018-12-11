#!/usr/bin/env python3
""" This file contains the np_cat function. """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ Concatenates two tensors along a specific axis. """
    mat_a = mat1[:]
    mat_b = mat2[:]
    return np.concatenate((mat_a, mat_b), axis=axis)
