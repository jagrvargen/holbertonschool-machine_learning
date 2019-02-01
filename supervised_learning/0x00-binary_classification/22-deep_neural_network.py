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

    def cost(self, Y, A):
        """ Calculates the cost of the model """
        cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / Y.shape[1])

        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions """
        self.forward_prop(X)

        A = self.__cache["A{}".format(self.__L)]
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Performs one pass of gradient descent on the neural network. """
        m = Y.shape[1]
        L = self.__L

        # Get the derivatives of the output layer
        dZ = cache["A{}".format(L)] - Y
        dW = dZ @ cache["A{}".format(L - 1)].T / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        self.__weights["W{}".format(L)] -= alpha * dW
        self.__weights["b{}".format(L)] -= alpha * db

        # Get the derivatives for every layer after layer 1 and before the output layer
        for l in range(L - 1, 0, -1):
            dZ = self.__weights["W{}".format(l+1)].T @ dZ * (cache["A{}".format(l)] * (1 - cache["A{}".format(l)]))
            dW = dZ @ cache["A{}".format(l - 1)].T / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights["W{}".format(l)] -= alpha * dW
            self.__weights["b{}".format(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the deep neural network """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(1, iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)

        prediction, cost = self.evaluate(X, Y)

        return prediction, cost
    
