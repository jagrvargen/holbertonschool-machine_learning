#!/usr/bin/env python3
""" This file contains the poly_derivative function. """


def poly_derivative(poly):
    """ Calculates the derivate of a polynomial. """
    if sum(poly) == 0:
        return [0]

    deriv = []

    for i in range(1, len(poly)):
        deriv.append(i * poly[i])

    return deriv
