#!/usr/bin/env python3
"""
Contains the function def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1))
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backward propagation over a convolutional layer of a neural
    network.
    dZ: numpy.ndarray (m, h_new, w_new, c_new) - Contains the partial
    derivatives with respect to the unactivated output of the convolutional
    layer.
    A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - Contains the outputs
    of the previous layer.
    W: numpy.ndrray (kh, kw, c_prev, c_new) - Contains the kernels for the
    convolution.
    b: numpy.ndarray (1, 1, 1, c_new) - Contains the biases applied to the
    convolution.
    padding: str - Determines what type of padding is used.
    stride: tuple (sh, sw) - The stride values of the convolution.

    Returns: The partial derivatives with respect to the previous layer,
    kernels, and biases.
    """
    # Create placeholders for derivatives
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Size of stride from forward prop
    s = stride[0]
