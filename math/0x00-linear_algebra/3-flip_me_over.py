#!/usr/bin/env python3
import numpy as np
def matrix_transpose(matrix):
    """ Returns the transpose of a matrix. """
    transpose = [[] for i in range(len(matrix[0]))]

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            transpose[j].append(matrix[i][j])
    
    return transpose
