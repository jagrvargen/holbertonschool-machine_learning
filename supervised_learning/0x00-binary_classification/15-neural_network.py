#!/usr/bin/env python3
"""
This file contains the class definition for a NeuralNetwork
"""
import numpy as np
import matplotlib.pyplot as plt


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
        Z1 = self.W1 @ X + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = self.W2 @ self.A1 + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model
        Y: numpy.ndarray shape=(1, m)
        A: numpy.ndarray shape=(1, m)
        """
        cost = -(np.sum(Y @ np.log(A.T) + (1 - Y) @ np.log(1 - A.T)) / Y.shape[1])
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the network's predictions
        X: np.ndarray shape=(nx, m)
        Y: np.ndarray shape=(1, m)
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.A2)
        prediction = np.where(self.A2 >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs gradient descent on the neural network
        X: numpy.ndarray shape=(nx, m)
        Y: numpy.ndarray shape=(1, m)
        A1: activated output of the hidden layer
        A2: activated output of the output layer
        """
        dZ2 = A2 - Y
        dW2 = dZ2 @ A1.T / X.shape[1]
        db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]

        dZ1 = self.W2.T @ dZ2 * (A1 * (1 - A1))
        dW1 = dZ1 @ X.T / X.shape[1]
        db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]

        self.__W2 = self.W2 - alpha * dW2
        self.__b2 = self.b2 - alpha * db2

        self.__W1 = self.W1 - alpha * dW1
        self.__b1 = self.b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neural network
        X: numpy.ndarray shape=(nx, m)
        Y: numpy.ndarray shape=(1, m)
        iterations: number of iterations to train over
        alpha: learning rate
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")

        if graph:
            x = np.arange(0, iterations + 1, step)
            y = np.empty((iterations // step + 1,))

        for i in range(iterations):
            self.forward_prop(X)

            if i% step == 0 or i == iterations:
                cost = self.cost(Y, self.A2)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    y[i // step] = cost
            
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)

        if graph:
            y[-1] = cost
            plt.plot(x, y)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.suptitle("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
