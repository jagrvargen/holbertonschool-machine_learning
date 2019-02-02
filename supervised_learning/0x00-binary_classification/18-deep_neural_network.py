#!/usr/bin/env python3
"""
Contains the class defintion for the DeepNeuralNetwork class
"""
import numpy as np


class DeepNeuralNetwork:
    """ Class definition for DeepNeuralNetwork """
    def __init__(self, nx, layers):
        """ Instantiates a DeepNeuralNetwork object """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        for val in layers:
            if type(val) is not int or val < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = dict()
        self.__weights = {"W1": np.random.randn(layers[0], nx) * np.sqrt(2/nx), "b1": 0}

        for i in range(1, len(layers)):
            self.__weights["W{}".format(i+1)] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/(layers[i-1]))
            self.__weights["b{}".format(i+1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Returns the number of layers of the deep neural network """
        return self.__L

    @property
    def cache(self):
        """ Returns the values stored in cache """
        return self.__cache

    @property
    def weights(self):
        """ Returns the values stored in the weights dictionary """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates forward propagation """
        self.__cache["A0"] = X

        for l in range(self.__L):
            Z = self.weights["W{}".format(l+1)] @ self.__cache["A{}".format(l)] + self.__weights["b{}".format(l+1)]
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A{}".format(l+1)] = A

        return A, self.cache
