#!/usr/bin/env python3
"""
Contains the function def evaluate(X, Y, save_path)
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of the neural network.
    X: numpy.ndarray - Contains the input data to evaluate.
    Y: numpy.ndarray - Contains the one-hot labels for X.
    save_path: string - Path to load the model from.

    returns: The network's prediction, accuracy, and loss
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)
        graph = tf.get_default_graph()
        x = graph.get_collection('x')[0]
        y = graph.get_collection('y')[0]
        y_pred = graph.get_collection('y_pred')[0]
        loss = graph.get_collection('loss')[0]
        accuracy = graph.get_collection('accuracy')[0]

        return sess.run((y_pred, loss, accuracy), {x: X, y: Y})
