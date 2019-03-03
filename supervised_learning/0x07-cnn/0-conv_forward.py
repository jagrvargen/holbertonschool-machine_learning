#!/usr/bin/env python3
"""
Contains the function def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1))
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural
    network.
    A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - Contains the output
    of the previous layer.
    W: numpy.ndarray (kh, kw, c_prev, c_new) - Contains the kernels for the
    convolution.
    b: numpy.ndarray (1, 1, 1, c_new) - Contains the biases to be applied to
    the convolution.
    activation: func - The activation function to be applied to the
    convolution.
    padding: str - Indicates the type of padding used.
    stride: tuple (sh, sw) - Contains the strides for the convolution.

    Returns: The output of the convolutional layer.
    """
    # Total samples
    m = A_prev.shape[0]

    # Height, width, and depth of input
    h_prev, w_prev, c_prev = A_prev.shape[1], A_prev.shape[2], A_prev.shape[3]

    # Height and width of kernel
    kh, kw = W.shape[0], W.shape[1]

    # Number of kernels
    c_new = W.shape[3]

    # Apply padding to input (or not)
    s = stride[0]
    if padding == "valid":
        p = (kh + s * (h_prev - 1) - h_prev) // 2
        A_prev = np.pad(A_prev, ((0, 0), (p, p), (p, p), (0,0)), 'constant', constant_values=0)
    else:
        p = 0

    # Calculate the size of the output and create a 4D tensor of 0s
    out_h = (h_prev - kh + 2 * p) // s + 1
    output = np.zeros((m, out_h, out_h, c_new))

    # Forward prop
    for layer in range(c_new):
        for i in range(out_h):
            for j in range(out_h):
                output[:,i,j,layer] = activation(np.sum(A_prev[:, i*s:i*s+kh, j*s:j*s+kw, :] * W[:,:,:,layer], axis=(1,2,3)) + b[:,:,:,layer])

    return output
