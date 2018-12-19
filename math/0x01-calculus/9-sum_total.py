#!/usr/bin/env python3
""" This file contains the function summation_i_squared. """


def summation_i_squared(n):
    """ Calculates the summation of i^2 from i = 1 to n """
    if not isinstance(n, int):
        return None

    s = range(1, n+1)
    return sum(list(map(lambda x: x**2, s)))
