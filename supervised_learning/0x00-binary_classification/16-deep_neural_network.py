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

        self.L = len(layers)
        self.cache = dict()
        self.weights = {"W1": np.random.randn(layers[0], nx) * np.sqrt(2/nx)}

        for i in range(1, len(layers)):
            self.weights["W{}".format(i+1)] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/(layers[i-1]))
            self.weights["b{}".format(i+1)] = np.zeros((layers[i], 1))
