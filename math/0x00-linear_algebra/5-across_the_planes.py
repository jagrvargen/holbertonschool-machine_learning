#!/usr/bin/env python3
""" This file contains the add_matrices2D function. """


def add_matrices2D(mat1, mat2):
    """ Performs element-wise addition of two 2D matrices. """
    if len(mat1[0]) != len(mat2[0]):
        return None

    summed_matrix = [[] for i in mat1]

    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            summed_matrix[i].append(mat1[i][j] + mat2[i][j])

    return summed_matrix
