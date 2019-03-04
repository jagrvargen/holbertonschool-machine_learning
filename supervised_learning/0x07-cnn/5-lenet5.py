#!/usr/bin/env python3
"""
Contains the function def lenet5(X)
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras.
    X: K.Input (m, 28, 28, 1) - Contains the input images for the network.

    Returns: A keras model
    """
    # Convolutional layer, 6 5x5 kernels, same padding
    x = K.layers.Conv2D(6, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(X)

    # Max pooling layer, 2x2 kernels, 2x2 strides
    x = K.layers.MaxPool2D((2, 2), (2, 2))(x)

    # Convolutional layer 16 5x5 kernels, valid padding
    x = K.layers.Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal')(x)

    # Max pooling layer, 2x2 kernels, 2x2 strides
    x = K.layers.MaxPool2D((2, 2), (2, 2))(x)

    # Flatten x for dense layers
    x = K.layers.Flatten()(x)

    # Fully connected layer, 120 nodes
    x = K.layers.Dense(120, activation='relu', kernel_initializer='he_normal')(x)

    # Fully connected layer, 84 nodes
    x = K.layers.Dense(84, activation='relu', kernel_initializer='he_normal')(x)

    # Softmax output layer, 10 nodes
    y_hat = K.layers.Dense(10, activation='softmax', kernel_initializer='he_normal')(x)

    model = K.Model(inputs=X, outputs=y_hat)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
