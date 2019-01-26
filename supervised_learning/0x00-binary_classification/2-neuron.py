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

        self.__W = np.random.normal(size=(nx, 1))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Returns the values of the weights vector """
        return self.__W

    @property
    def b(self):
        """ Returns the value of the bias neuron """
        return self.__b

    @property
    def A(self):
        """ Returns the value of the activations output """
        return self.__A

    def forward_prop(self, X):
        """ Calulates forward propagation of the neuron """
        z = np.matmul(self.__W.T, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A
