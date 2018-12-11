#!/usr/bin/env python3


def matrix_shape(matrix):
    """ Calculates the shape of an n-dimensional matrix. """
    shape = []
    m = matrix[:]

    while not isinstance(m, int):
        shape.append(len(m))
        m = m.pop()

    return shape
