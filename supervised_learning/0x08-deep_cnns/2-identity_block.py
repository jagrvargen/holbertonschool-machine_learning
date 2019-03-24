#!/usr/bin/env python3
"""
Contains the function def identity_block(A_prev, filters)
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block
    A_prev: K.Tensor - Contains the outputs of the previous layer.
    filters: (tuple) - Contains the number of filters for each convolution.
    """
    F11, F3, F12 = filters
    df = "channels_last"

    # 1x1 conv
    F11 = K.layers.Conv2D(F11, (1, 1), data_format=df, activation="relu")(A_prev)
    F11 = K.layers.BatchNormalization()(F11)

    # 3x3 conv
    F3 = K.layers.Conv2D(F3, (3, 3), data_format=df, padding="same", activation="relu")(F11)
    F3 = K.layers.BatchNormalization()(F3)

    # 1x1 conv
    F12 = K.layers.Conv2D(F12, (1, 1), data_format=df, activation="relu")(F3)
    F12 = K.layers.BatchNormalization()(F12)

    return K.layers.Add()([A_prev, F12])
