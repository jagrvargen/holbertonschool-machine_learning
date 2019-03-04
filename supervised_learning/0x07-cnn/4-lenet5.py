#!/usr/bin/env python3
"""
Contains the function def lenet5(x, y)
"""
import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture
    x: tf.placeholder (m, 28, 28, 1) - Contains the input images for the
    network.
    y: tf.placeholder (m, 10) - Contains the one-hot labels for the network

    Returns: A tensor containing the softmax activated output, an Adam training
    op, a tensor containing the loss, and a tensor containg the accuracy.
    """
    # First convolutional layer, 6 5x5 kernels, same padding
    conv_layer_1 = tf.layers.Conv2D(6, (5, 5), padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'), name='conv_layer_1')(x)

    # Max pooling layer, 2x2 kernels, 2x2 strides
    pooling_layer_1 = tf.layers.MaxPooling2D((2, 2), (2, 2), name='pooling_layer_1')(conv_layer_1)

    # Convolutional layer, 16 5x5 kernels, valid padding
    conv_layer_2 = tf.layers.Conv2D(16, (5, 5), padding='valid', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'), name='conv_layer_2')(pooling_layer_1)

    # Max pooling layer, 2x2 kernels, 2x2 strides
    pooling_layer_2 = tf.layers.MaxPooling2D((2, 2), (2, 2), name='pooling_layer')(conv_layer_2)

    # Flatten output for fully connected layers
    dense_input = tf.layers.Flatten()(pooling_layer_2)

    # Fully connected layer, 120 nodes
    dense_layer_1 = tf.layers.Dense(120, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'), name='dense_layer_1')(dense_input)

    # Fully connected layer, 84 nodes
    dense_layer_2 = tf.layers.Dense(84, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'), name='dense_layer_2')(dense_layer_1)

    # Fully connected output layer, 10
    y_hat = tf.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'), name='output_layer')(dense_layer_2)

    # Calculate loss
    loss = tf.losses.softmax_cross_entropy(y, y_hat)

    # Create optimizer
    op = tf.train.AdamOptimizer().minimize(loss)

    # Calculate accuracy
    true_vals = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
    accuracy = tf.reduce_mean(tf.cast(true_vals, tf.float32))

    return y_hat, op, loss, accuracy
