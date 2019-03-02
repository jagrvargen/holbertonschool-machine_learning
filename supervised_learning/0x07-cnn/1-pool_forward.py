#!/usr/bin/env python3
"""
Contains the function def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max')
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.
    A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - Contains the output
    of the previous layer.
    kernel_shape: tuple (kh, kw) - Contains the height and width of the
    kernel used for pooling.
    stride: tuple (sh, sw) - Contains the horizontal and vertical stride
    values.
    mode: str - Determines whether to use max or average pooling.

    Returns: The output of the pooling layer.
    """
    # Number of samples
    m = A_prev.shape[0]

    # Height and depth of input
    h_prev, c_prev = A_prev.shape[1], A_prev.shape[3]

    # 1 dimension of kernel
    k = kernel_shape[0]

    # Stride
    s = stride[0]

    # 1 dimension of output
    out_d = (h_prev - k) // s + 1

    output = np.zeros((m, out_d, out_d, c_prev))

    for depth in range(c_prev):
        for height in range(out_d):
            for width in range(out_d):
                if mode == 'max':
                    output[:, height, width, depth] = np.amax(A_prev[:, height*s:height*s+k, width*s:width*s+k, depth], axis=(1, 2))
                elif mode == 'avg':
                    output[:, height, width, depth] = np.sum(A_prev[:, height*s:height*s+k, depth], axis=(1, 2, 3)) / (k ** 2)

    return output
