#!/usr/bin/env python3
""" This function contains the  cat_matrices2D function. """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two 2D matrices """
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None

    if axis == 1 and len(mat1) != len(mat2):
        return None

    if axis != 0 and axis != 1:
        return None

    concat = []

    if axis == 0:
        concat.extend(mat1)
        concat.extend(mat2)

    else:
        for i in range(len(mat1)):
            concat.append([])
            concat[i].extend(mat1[i])
            concat[i].extend(mat2[i])

    return concat
