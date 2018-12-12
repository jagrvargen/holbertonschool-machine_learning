#!/usr/bin/env python3
""" This file contains the np_slice function. """


def np_slice(matrix, axes={}):
    """ Slices a matrix along specific axes. """
    zero = eval('slice' + str(axes[0])) if 0 in axes else\
        slice(None, None, None)
    one = eval('slice' + str(axes[1])) if 1 in axes else\
        slice(None, None, None)
    two = eval('slice' + str(axes[2])) if 2 in axes else\
        slice(None, None, None)

    if len(matrix.shape) == 1:
        return matrix[zero]
    if len(matrix.shape) == 2:
        return matrix[zero, one]
    elif len(matrix.shape) == 3:
        return matrix[zero, one, two]
    else:
        return matrix
