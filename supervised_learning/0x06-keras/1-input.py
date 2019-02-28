#!/usr/bin/env python3
"""
Contains the function def build_model(nx, layers, activations, lambtha, keep_prob)
"""
import tensorflow as tf
import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.
    nx: int - The number of input features to the network.
    layers: list - Contains the number of nodes in each layer of the network.
    activations: list - Contains the activation functions used for each layer
    of the network.
    lambtha: float - The L2 regularization parameter.
    keep_prob: float - The probability that a node will be kept for dropout.

    Returns: The Keras model.
    """
    inputs = keras.Input(shape=(nx,))

    x = keras.layers.Dense(layers[0], activation=activations[0], kernel_regularizer=keras.regularizers.l2(l=lambtha))(inputs)
    x = keras.layers.Dropout(1 - keep_prob)(x)

    for i in range(1, len(layers)):
        x = keras.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=keras.regularizers.l2(l=lambtha))(x)
        if i < len(layers) - 1:
            x = keras.layers.Dropout(1 - keep_prob)(x)

    model = keras.Model(inputs=inputs, outputs=x)
            
    return model
