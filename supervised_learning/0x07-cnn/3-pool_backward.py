#!/usr/bin/env python3
"""
Contains the function def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max')
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backpropagation over a pooling layer of a neural network.
    dA: numpy.ndarray (m, h_new, w_new, c_new) - Contains the partial
    derivatives with respect to the output of the pooling layer.
    A_prev: numpy.ndarray (m, h_prev, w_prev, c) - Contains the output of the
    previous layer.
    kernel_shape: tuple (kh, kw) - Contains the height and width of the
    kernel used for pooling.
    stride: tuple (sh, sw) - Contains the strides used for pooling.
    mode: str - Indicates whether max or average pooling was used.

    Returns: The partial derivatives with respect to the previous layer.
    """
    # dA shape
    m, h_new, w_new, c_new = dA.shape[0], dA.shape[1], dA.shape[2], dA.shape[3]

    # Kernel height and width
    kh, kw = kernel_shape[0], kernel_shape[1]

    # Stride
    s = stride[0]

    dA_prev = np.zeros(A_prev.shape)

    for sample in range(m):
        for depth in range(c_new):
            for height in range(h_new):
                for width in range(w_new):
                    chunk = A_prev[sample, height*s:height*s+kw, width*s:width*s+kw, depth]
                    if mode == 'max':
                        mask = (chunk == np.max(chunk))
                        dA_prev[sample, height*s:height*s+kh, width*s:width*s+kw, depth] += mask * dA[sample, height, width, depth]
                    elif mode == 'avg':
                        dA_prev[sample, height*s:height*s+kh, width*s:width*s+kw, depth] += dA[sample, height, width, depth] / (kh * kw)

    return dA_prev
