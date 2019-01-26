#!/usr/bin/env python3
"""
Neuron class definition
"""
import numpy as np


class Neuron:
    """ Neuron Class """
    def __init__(self, nx):
        """ Creates a Neuron object instance """
        if not isinstance(nx, int):
            raise TypeError("nx must me an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.W = np.random.normal(size=(nx, 1))
        self.b = 0
        self.A = 0
