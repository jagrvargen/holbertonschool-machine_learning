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
    # Number of samples
    m = A_prev.shape[0]
    
    # Create placeholders for derivatives
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Size of stride from forward prop
    s = stride[0]

    # Retrieve height and width of kernel
    kh, kw = W.shape[0], W.shape[1]

    # Height, width, and depth of dZ
    h_new, w_new, c_new = dZ.shape[1], dZ.shape[2], dZ.shape[3]

    # Retrieve height, width, and depth of previous input layer
    h_prev, w_prev, c_prev = A_prev.shape[1], A_prev.shape[2], A_prev.shape[3]

    # Padding
    p = (kh + s * (h_prev - 1) - h_prev) // 2
    
    if padding == "same":
        p = (h_prev - kh + 2 * p) // s + 1
        dA_prev = np.pad(dA_prev, ((0, 0), (p, p), (p, p), (0, 0)), 'constant', constant_values=0)
        A_prev_pad = np.pad(A_prev, ((0, 0), (p, p), (p, p), (0, 0)), 'constant', constant_values=0)
    else:
       p = 0
       A_prev_pad = A_prev

    for sample in range(m):
        for depth in range(c_new):
            for height in range(h_new):
                for width in range(w_new):
                    dA_prev[sample, height*s:height*s+kh, width*s:width*s+kw, :] += W[:, :, :, depth] * dZ[sample, height, width, depth]
                    dW[:, :, :, depth] += A_prev_pad[sample, height*s:height*s+kh, width*s:width*s+kw, :] * dZ[sample, height, width, depth]
                    db[:, :, :, depth] += dZ[sample, height, width, depth]

    if p:
        return dA_prev[:, p:-p, p:-p, :], dW, db
    return dA_prev, dW, db
