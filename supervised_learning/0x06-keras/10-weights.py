#!/usr/bin/env python3
"""
Contains:
   save_weights(network, filename, save_format='h5')
   load_weights(network, filename)
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves a trained network's weights.
    network: keras.Model - A trained Keras model.
    filename: str - The path to which to save the weights.
    save_format: str - The file format  under which tosave the weights.

    Returns: Nothing
    """
    network.save_weights(filename, save_format=save_format)

def load_weights(network, filename):
    """
    Loads a trained model's weights.
    network: keras.Model - A Keras model.
    filename: str - The path from which to load the weights.

    Returns: Nothing.
    """
    network.load_weights(filename)
