#!/usr/bin/env python3
""" This file contains the poly_derivative function. """


def poly_derivative(poly):
    """ Calculates the derivate of a polynomial. """
    if not poly:
        return None

    deriv = []

    for i, c in enumerate(poly):
        deriv.append(i * c)

    return deriv
