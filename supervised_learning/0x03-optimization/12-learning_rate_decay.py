#!/usr/bin/env python3
"""
Contains the function def learning_rate_decay(alpha, decay_rate, global_step, decay_step)
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation using inverse time decay.
    alpha: float - The initial learning rate.
    decay_rate: float - The weight which determines the rate at which
    the learning rate decays.
    global_step: int - The number of passes of gradient descent that have
    elapsed.
    decay_step: int - The number of passes of gradient descent that should
    occur before the learning rate is further decayed.

    Returns: The learning rate decay operation.
    """
    return tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, staircase=True)
