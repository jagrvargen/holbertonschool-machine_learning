#!/usr/bin/env python3
"""
Contains the function def inception_network()
"""
import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the Inception Network
    """
    df = "channels_last"  # Data format
    
    # Input
    X = K.Input((224, 224, 3), dtype="float32")

    # 1st Conv. layer
    C1 = K.layers.Conv2D(64, (7, 7), data_format=df, strides=(2, 2), padding="same", activation="relu")(X)
    MP1 = K.layers.MaxPooling2D((3, 3), (2, 2), padding="same", data_format=df)(C1)

    # 2nd Conv. layer
    C2 = K.layers.Conv2D(192, (3, 3), data_format=df, strides=(1, 1), padding="same", activation="relu")(MP1)
    MP2 = K.layers.MaxPooling2D((3, 3), (2, 2), padding="same", data_format=df)(C2)

    # First Inception
    IB1 = inception_block(MP2, (64, 96, 128, 16, 32, 32))

    # Second Inception
    IB2 = inception_block(IB1, (128, 128, 192, 32, 96, 63))

    # Max Pool
    MP3 = K.layers.MaxPooling2D((3, 3), (2, 2), padding="same", data_format=df)(IB2)

    # 3rd Inception
    IB3 = inception_block(MP3, (192, 96, 208, 16, 48, 64))

    # 4th Inception
    IB4 = inception_block(IB3, (160, 112, 224, 24, 64, 64))

    #5th Inception
    IB5 = inception_block(IB4, (128, 128, 256, 24, 64, 64))

    # 6th Inception
    IB6 = inception_block(IB5, (112, 144, 288, 32, 64, 64))

    # 7th Inception
    IB7 = inception_block(IB6, (256, 160, 320, 32, 128, 128))

    # Max Pool
    MP4 = K.layers.MaxPooling2D((3, 3), (2, 2), padding="same", data_format=df)(IB7)

    # 8th Inception
    IB8 = inception_block(MP4, (256, 160, 320, 32, 128, 128))

    # 9th Inception
    IB9 = inception_block(IB8, (384, 192, 384, 48, 128, 128))

    # Avg. Pool
    AP1 = K.layers.AveragePooling2D((7, 7), (1, 1), padding="same", data_format=df)(IB9)

    # Dropout layer
    D1 = K.layers.Dropout(.4)(AP1)

    # Dense layer
    Y_hat = K.layers.Dense(1000, activation="softmax")(D1)

    return K.models.Model(inputs=X, outputs=Y_hat)
