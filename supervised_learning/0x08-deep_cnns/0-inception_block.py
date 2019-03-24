#!/usr/bin/env python3
"""
Contains the function def inception_block(A_prev, filters)
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block
    A_prev: K.Tensor - Output of previous layer
    filters: tuple - Contains # of filters for each convolution
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    df = "channels_last"  # Data format

    F1 = K.layers.Conv2D(F1, (1, 1), data_format=df, activation="relu")(A_prev)

    F3R = K.layers.Conv2D(F3R, (1, 1), data_format=df, activation="relu")(A_prev)
    F3 = K.layers.Conv2D(F3, (3, 3), data_format=df, padding="same", activation="relu")(F3R)

    F5R = K.layers.Conv2D(F5R, (1, 1), data_format=df, activation="relu")(A_prev)
    F5 = K.layers.Conv2D(F5, (5, 5), data_format=df, padding="same", activation="relu")(F5R)

    MP3 = K.layers.MaxPooling2D((3, 3), (1, 1), padding="same", data_format=df)(A_prev)
    FPP = K.layers.Conv2D(FPP, (1, 1), data_format=df, activation="relu")(MP3)

    return K.layers.Concatenate(axis=-1)([F1, F3, F5, FPP])
