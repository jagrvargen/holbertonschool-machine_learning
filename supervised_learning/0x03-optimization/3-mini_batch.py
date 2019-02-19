#!/usr/bin/env python3
"""
Contains the function def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt")
"""
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    Trains a neural network using mini-batch gradient descent
    X_train: numpy.ndarray (m, 784) - Where m is the number of data points and
    784 is the number of input features.
    Y_train: numpy.ndarray (m, 10) - Where m is the number of data points and
    10 is the number of classes.
    X_valid: numpy.ndarray (m, 784) - Cross validation inputs.
    Y_valid: numpy.ndarray (m, 10) - Cross validation labels.
    batch_size: int - The number of data points per batch.
    epochs: int - The number of times the model is trained on the full dataset.
    load_path: str - The path from which to load a trained model.
    save_path: str - The path to which to save a trained model.

    Returns: The path to which the model was saved.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        steps = X_train.shape[0] // batch_size + 1

        for epoch in range(epochs):
            print("After {} epochs:".format(epoch))
            
            train_cost, train_accuracy = sess.run([loss, accuracy], {x: X_train, y: Y_train})
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))

            valid_cost, valid_accuracy = sess.run([loss, accuracy], {x: X_valid, y: Y_valid})
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            for step in range(steps):
                X_train, Y_train = shuffle_data(X_train, Y_train)
                if step < steps - 1:
                    end = step + batch_size + 1
                else:
                    end = X_train.shape[0] % batch_size + 1
                sess.run(train_op, {x: X_train[step:end], y: Y_train[step:end]})
                if step % 100 == 0:
                    train_cost, train_accuracy = sess.run([loss, accuracy], {x: X_train[step:end], y: Y_train[step:end]})
                    print("\tStep: {}".format(step))
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
