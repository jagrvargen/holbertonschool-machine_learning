#!/usr/bin/env python3
"""
Contains the function def create_placeholders(nx, classes)
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders x and y
    nx: int - The number of feature columns in the data
    classes: int - The number of classes in the classifier
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
