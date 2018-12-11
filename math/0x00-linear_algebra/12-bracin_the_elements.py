#!/usr/bin/env python3
""" This file contains the np_elementwise function. """


def np_elementwise(mat1, mat2):
    """ 
       Performs element-wise addition, subtraction, multiplication, and
       division on two n-dimensional tensors.
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return add, sub, mul, div
