#!/usr/bin/env python3
""" This file contains the np_slice function. """


def np_slice(matrix, axes={}):
    """ Slices a matrix along specific axes. """
    s0 = eval('slice' + str((axes[0]))) if 0 in axes else slice(None)
    s1 = eval('slice' + str(axes[1])) if 1 in axes else slice(None)
    s2 = eval('slice' + str(axes[2])) if 2 in axes else slice(None)

    if len(matrix.shape) == 1:
        return matrix[s0]
    if len(matrix.shape) == 2:
        return matrix[s0, s1]
    elif len(matrix.shape) == 3:
        return matrix[s0, s1, s2]
    else:
        return matrix
