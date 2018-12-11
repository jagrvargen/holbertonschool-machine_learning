#!/usr/bin/env python3
""" This file contains the matrix_transpose function. """


def matrix_transpose(matrix):
    """ Returns the transpose of a matrix. """
    transpose = [[] for i in range(len(matrix[0]))]

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            transpose[j].append(matrix[i][j])

    return transpose
