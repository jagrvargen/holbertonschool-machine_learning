#!/usr/bin/env python3
""" This file contains the np_slice function. """


def np_slice(matrix, axes={}):
    """ Slices a matrix along specific axes. """

    s0 = eval('slice' + str((axes[0]))) if 0 in axes else slice(None)
    s1 = eval('slice' + str(axes[1])) if 1 in axes else slice(None)
    s2 = eval('slice' + str(axes[2])) if 2 in axes else slice(None)

    
    if len(matrix.shape) == 1:
        s = matrix[s0]
        return s
    if len(matrix.shape) == 2:
        s = matrix[s0, s1]
        return s
    elif len(matrix.shape) == 3:
        s = matrix[s0, s1, s2]
        return s
    else:
        s = matrix
        return s
