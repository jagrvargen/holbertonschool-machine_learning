#!/usr/bin/env python3
"""
Contains the function train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt")
"""
import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Build, trains, and saves the neural network classifier.
    X_train: numpy.ndarray - Contains the training input data.
    Y_train: numpy.ndarray - Contains the training labels.
    X_valid: numpy.ndarray - Contains the validation input data.
    Y_valid: numpy.ndarray - Contains the validation labels.
    layer_sizes: list - Contains the number of nodes in each layer of the
    network.
    activations: list - Contains the activation functions for each layer of
    the network.
    alpha: float - The learning rate.
    iterations: int - The number of iterations to train over.
    save_path: string - The path to save the model to.

    returns: string - The path to which the model was saved.
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    
    y_pred = forward_prop(x, x.shape[1], layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(iterations):
            if i % 100 == 0:
                print("After {} iterations".format(i))
                train_cost, train_accuracy = sess.run([loss, accuracy], {x: X_train, y: Y_train})
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                valid_cost, valid_accuracy = sess.run([loss, accuracy], {x: X_valid, y: Y_valid})
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))
            sess.run(train_op, {x: X_train, y: Y_train})

        save_path = saver.save(sess, save_path)

        return save_path
