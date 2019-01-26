#!/usr/bin/env python3
"""
Class definition for Neuron with private instance attributes
"""
import numpy as np


class Neuron:
    """ Neuron class """
    def __init__(self, nx):
        """ Instantiates a Neuron object """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(size=nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
