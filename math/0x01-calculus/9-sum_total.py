#!/usr/bin/env python3
""" This file contains the function summation_i_squared. """


def summation_i_squared(n):
    """ Calculates the summation of i^2 from i = 1 to n """
    if not isinstance(n, int) or n < 1:
        return None
    if n == 1:
        return 1

    return n**2 + summation_i_squared(n - 1)
