#!/usr/bin/env python3
"""
Contains the function def build_model(nx, layers, activations, lambtha, keep_prob)
"""
import tensorflow as tf
import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras Library.
    nx: int - The number of input features to the network.
    layers: list - Contains the number of nodes in each layer of the network.
    activations: list - Contains the activation functions used for each layer
    of the network.
    lambtha: float - The L2 regularization parameter.
    keep_prob: float - The probability that a node will be kept for dropout.

    Returns: The Keras model.
    """
    model = tf.keras.Sequential()

    model.add(keras.layers.Dense(layers[0], activation=activations[0], input_shape=(nx,), kernel_regularizer=keras.regularizers.l2(l=lambtha)))
    model.add(keras.layers.Dropout(1 - keep_prob))

    for i in range(1, len(layers)):
        model.add(keras.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=keras.regularizers.l2(l=lambtha)))
        if i < len(layers) - 1:
            model.add(keras.layers.Dropout(1 - keep_prob))

    return model
