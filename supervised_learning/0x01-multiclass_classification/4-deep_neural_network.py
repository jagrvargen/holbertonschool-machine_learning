#!/usr/bin/env python3
"""
Contains the class defintion for the DeepNeuralNetwork class
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ Class definition for DeepNeuralNetwork """
    def __init__(self, nx, layers, activation='sig'):
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

        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
        self.__L = len(layers)
        self.__cache = dict()
        self.__weights = {"W1": np.random.randn(layers[0], nx) * np.sqrt(2/nx), "b1": 0}

        for i in range(1, len(layers)):
            self.__weights["W{}".format(i+1)] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/(layers[i-1]))
            self.__weights["b{}".format(i+1)] = np.zeros((layers[i], 1))

    @property
    def activation(self):
        """ Returns the type of activation functions of the hidden layers """
        return  self.__activation

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
        L = self.__L
        self.__cache["A0"] = X

        for l in range(L - 1):
            Z = self.weights["W{}".format(l+1)] @ self.__cache["A{}".format(l)] + self.__weights["b{}".format(l+1)]
            if self.activation == 'sig':
                A = 1 / (1 + np.exp(-Z))
            elif self.activation == 'tanh':
                A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            self.__cache["A{}".format(l+1)] = A

        Z = self.weights["W{}".format(L)] @ self.__cache["A{}".format(L-1)] + self.__weights["b{}".format(L)]
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        self.__cache["A{}".format(L)] = A

        return A, self.cache

    def cost(self, Y, A):
        """ Calculates the cost of the model """
        m = Y.shape[1]
        loss = -(np.sum(Y * np.log(A), axis=0))
        cost = np.sum(loss) / m
        
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions """
        self.forward_prop(X)

        A = self.__cache["A{}".format(self.__L)]
        cost = self.cost(Y, A)

        predicted_values = np.argmax(A)
        num_classes = np.max(predicted_values) + 1
        prediction = np.eye(num_classes)[predicted_values]

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
            A = cache["A{}".format(l)]
            if self.activation == 'sig':
                dZ = self.__weights["W{}".format(l+1)].T @ dZ * (A * (1 - A))
            elif self.activation == 'tanh':
                dZ  = self._weights["W{}".format(l+1)].T @ dZ * (1 - ((np.exp(A) - np.exp(-A) / (np.exp(A) + np.exp(-A)))) ** 2)
            dW = dZ @ cache["A{}".format(l - 1)].T / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights["W{}".format(l)] -= alpha * dW
            self.__weights["b{}".format(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ Trains the deep neural network """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if graph:
            x = np.arange(0, iterations + 1, step)
            y = np.empty((iterations // step + 1,))

        for i in range(iterations + 1):
            self.forward_prop(X)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.__cache["A{}".format(self.__L)])
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    y[i // step] = cost
            
            self.gradient_descent(Y, self.cache, alpha)

        if graph:
            y[-1] = cost
            plt.plot(x, y)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.suptitle("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves a DeepNeuralNetwork model in pickle format """
        if filename[-4:] != ".pkl":
            filename += ".pkl"

        with open(filename, 'wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork model """
        with open(filename, 'rb') as fd:
            model = pickle.load(fd)

        return model
