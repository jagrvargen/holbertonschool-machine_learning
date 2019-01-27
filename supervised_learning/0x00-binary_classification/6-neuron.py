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
        self.__b = np.zeros((1, 1))
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

    def cost(self, Y, A):
        """ Logistic regression cost function """
        cost = -(np.sum(Y @ np.log(A.T) + (1 - Y) @ np.log(1 - A.T)) / Y.shape[1])
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron's predictions """
        self.forward_prop(X)        
        cost = self.cost(Y, self.A)
        prediction = np.where(self.A >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        dz = A - Y
        dw = X @ dz.T / X.shape[1]
        db = np.sum(dz.T) / X.shape[1]

        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neuron """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            print(i)
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A, alpha)

        return self.evaluate(X, Y)
