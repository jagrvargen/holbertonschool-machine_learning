#!/usr/bin/env python3
"""
Contains:
   def save_config(network, filename)
   def load_config(filename)
"""
import json
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.
    network: keras.Model - The model to be saved.
    filename: str - The path to which to save the model.

    Returns: Nothing
    """
    with open(filename, 'w+') as fp:
        json.dump(network.to_json(), fp)

        
def load_config(filename):
    """
    Loads a model's configration from a JSON file.
    filename: str - The path from which to load the model.

    Returns: The loaded model.
    """
    with open(filename, 'r+') as fp:
        json_string = json.load(fp)
        return K.models.model_from_json(json_string)
