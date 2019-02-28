#!/usr/bin/env python3
"""
Contains:
   def save_model(network, filename)
   def load_model(filename)
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves a trained Keras model.
    network: keras.Model - A trained keras model.
    filename: str - The path to which to save the trained model.

    Returns: Nothing.
    """
    network.save(filename)


def load_model(filename):
    """
    Loads a trained Keras model.
    filename: str - The path from which to load the trained model.

    Returns: The loaded Keras model.
    """
    return K.models.load_model(filename)
