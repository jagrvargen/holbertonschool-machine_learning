#!/usr/bin/env python3
""" This file contains the function summation_i_squared. """


def summation_i_squared(n):
    """ Calculates the summation of i^2 from i = 1 to n """
    if not isinstance(n, int) and not isinstance(n, float):
        return None
    
    return int((n * (n + 1) * (2 * n + 1)) / 6)
