#!/usr/bin/env python3
"""
Contains the function def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt')
"""
import numpy as np
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model using Adam optimization,
    mini-batch gradient descent, learning rate decay, and batch normalization.
    Data_train: tuple (numpy.ndarray, numpy.ndarray) - Contains the training
    inputs and labels.
    and training labels.
    Data_valid: tuple (numpy.ndarray, numpy.ndarray) - Contains the validation
    inputs and labels.
    and validation labels.
    layers: list - The number of nodes in each layer of the network.
    activations: list - The activation functions for each layer of the network.
    alpha: float - The learning rate.
    beta1: float - The weight for the first moment of Adam optimization.
    beta2: float - The weight for the second moment of Adam optimization
    epsilon: float - Used to prevent division by zero.
    decay_rate: int - The inverse time decay rate.
    batch_size: int - The size of each mini-batch.
    epochs: int - The number of training epochs.
    save_path: str - The path to which to save the model.

    Returns: The path to which the model was saved.
    """
    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]
    
    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    a = x
    
    # Add subsequent layers with activated outputs from previous layers as inputs
    for i in range(len(layers)):
        layer = tf.layers.Dense(layers[i], kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), name="layer")
        z = layer(a)
        
        if i < len(layers) - 1:
            gamma = tf.Variable(tf.constant(1.0, shape=[layers[i]]), name="gamma", trainable=True)
            beta = tf.Variable(tf.constant(0.0, shape=[layers[i]]), name="beta", trainable=True)
            mean, variance = tf.nn.moments(z, axes=0)
            z_norm = tf.nn.batch_normalization(z, mean, variance, beta, gamma, 1e-8)
            a = activations[i](z_norm)
        else:
            a = z

    # Add the predictied values to the tf.Collection
    y_pred = a
    tf.add_to_collection('y_pred', y)

    # Calculate loss
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    # Initialize learning rate decay
    global_step = tf.Variable(0, trainable=False)
    decay_step = X_train.shape[0] // batch_size
    if X_train.shape[0] % batch_size == 0:
        decay_step += 1
    alpha = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, staircase=True)

    # Train using the Adam optimizer
    train_op = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss, global_step=global_step)
    tf.add_to_collection('train_op', train_op)

    # Calculate accuracy
    true_vals = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(true_vals, tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    # Create variables initializer operation
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        for epoch in range(epochs):
            print("After {} epochs:".format(epoch))

            train_cost, train_accuracy = sess.run([loss, accuracy], {x: X_train, y: Y_train})
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))

            valid_cost, valid_accuracy = sess.run([loss, accuracy], {x: X_valid, y: Y_valid})
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            permutation = np.random.permutation(X_train.shape[0])
            X_shuffle = X_train[permutation]
            Y_shuffle = Y_train[permutation]

            for step in range(0, X_train.shape[0], batch_size):
                if step < X_train.shape[0] - 1:
                    end = step + batch_size + 1
                else:
                    end = X_train.shape[0] % batch_size + 1
                sess.run(train_op, {x: X_shuffle[step:end], y: Y_shuffle[step:end]})
                if step // batch_size % 100 == 0:
                    train_cost, train_accuracy = sess.run([loss, accuracy], {x: X_shuffle[step:end], y: Y_shuffle[step:end]})
                    print("\tStep: {}".format(step // batch_size))
                    print("\t\tCost: {}".format(train_cost))
                    print("\t\tAccuracy: {}".format(train_accuracy))

        print("After {} epochs:".format(epochs))
        train_cost, train_accuracy = sess.run([loss, accuracy], {x: X_train, y: Y_train})
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        valid_cost, valid_accuracy = sess.run([loss, accuracy], {x: X_valid, y: Y_valid})
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        saver = tf.train.Saver()
        return saver.save(sess, save_path)
