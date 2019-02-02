
#!/usr/bin/env python3
"""
Class definition for Neuron with private instance attributes
"""
import numpy as np
import matplotlib.pyplot as plt


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
        self.__Z = 0

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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ Trains the neuron """
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
            y = np.empty((iterations // step + 1, 4))
            
        for i in range(iterations + 1):
            self.forward_prop(X)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    y[i // step] = cost
                        
            self.gradient_descent(X, Y, self.A, alpha)                        

        if graph:
            y[-1] = cost
            plt.plot(x, y)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.suptitle("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
