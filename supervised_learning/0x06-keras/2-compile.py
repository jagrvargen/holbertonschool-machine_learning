#!/usr/bin/env python3
"""
Contains the function def optimize_model(network, alpha, beta1, beta2)
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Compiles a Keras model with Adam optimization.
    network: keras.Model - The model to optimize.
    alpha: float - The learning rate.
    beta1: float - The first parameter of the Adam optimizer.
    beta2: float - The second parameter of the Adam optimizer.

    Returns: Nothing.
    """
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2), loss='categorical_crossentropy', metrics=['accuracy'])
