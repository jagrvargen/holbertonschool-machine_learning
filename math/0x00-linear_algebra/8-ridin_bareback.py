#!/usr/bin/env python3
""" This file contains the mat_mul function. """


def mat_mul(mat1, mat2):
    """ Performs matrix multiplication on 2D matrices. """
    if len(mat1[0]) != len(mat2):
        return None

    result = [[] for i in mat1]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            total = 0
            for k in range(len(mat1[0])):
                total += mat1[i][k] * mat2[k][j]
            result[i].append(total)

    return result
