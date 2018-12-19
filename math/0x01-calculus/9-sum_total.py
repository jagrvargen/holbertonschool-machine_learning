#!/usr/bin/env python3
""" This file contains the function summation_i_squared. """


def summation_i_squared(n):
    """ Calculates the summation of i^2 from i = 1 to n """
    sum = 0
    for i in range(1, n+1):
        sum += i**2

    return sum
