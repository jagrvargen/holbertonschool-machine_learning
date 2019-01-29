#!/usr/bin/env python3
"""
This file contains the class definition for a NeuralNetwork
"""
import numpy as np


class NeuralNetwork:
    """ Class definition for NeuralNetwork """
    def __init__(self, nx, nodes):
        """ Instantiates a NeuralNetwork object. """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer values
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Output layer values
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Returns the weights vector of the hidden layer """
        return self.__W1

    @property
    def b1(self):
        """ Returns the bias value for the hidden layer """
        return self.__b1

    @property
    def A1(self):
        """ Returns the hidden layer's activated outputs """
        return self.__A1

    @property
    def W2(self):
        """ Returns the weights vector of the output layer """
        return self.__W2

    @property
    def b2(self):
        """ Returns the bias value for the hidden layer """
        return self.__b2

    @property
    def A2(self):
        """ Returns the output layer's activated outputs """
        return self.__A2

    def forward_prop(self, X):
        """
        Performs forward propagation on through the neural network
        X: numpy.ndarray shape=(nx, m)
        """
        z1 = self.W1 @ X + self.b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        z2 = self.W2 @ self.A1 + self.b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y: numpy.ndarray shape=(1, m)
        A: numpy.ndarray shape=(1, m)
        """
        cost = -(np.sum(Y @ np.log(A.T) + (1 - Y) @ np.log(1 - A.T)) / Y.shape[1])
        return cost
