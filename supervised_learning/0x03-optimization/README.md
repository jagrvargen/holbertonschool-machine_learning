# Supervised Learning - Optimization

### 0-norm_constants.py - Contains the function definition normalization_constants(X) which calculates the normalization constants of a matrix.

### 1-normalize.py - Contains the function definition normalize(X, m, v, epsilon) which normalizes a matrix.

### 2-shuffle_data.py - Contains the function definition shuffle_data(X, Y) which shuffles the data points in two matrices in parallel.

### 3-mini_batch.py - Contains the function definition train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt") which trains a neural network using mini-batch gradient descent.

### 4-moving_average.py - Contains the function definition moving_average(data, beta) which calculates the weighted moving average of a data set.

### 5-momentum.py - Contains the function definition update_variables_momentum(alpha, beta1, var, grad, v) which updates a variable using the gradient descent with momentum optimization algorithm.

### 6-momentum.py - Contains the function definition create_momentum_op(loss, alpha, beta1) which creates a training operation for a neural network in TensorFlow using the gradient descent with momentum optimization algorithm.

### 7-RMSProp.py - Contains the function definition update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s) which updates a variable using the RMSProp optimization algorithm.

### 8-RMSProp.py - Contains the function definition create_RMSProp_op(loss, alpha, beta2, epsilon) which creates the training operation for a neural network in TensorFlow using the RMSProp optimization algorithm.

### 9-Adam.py - Contains the function definition update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t) updates a variable using the Adam optimization algorithm.

### 10-Adam.py - Contains the function definition create_Adam_op(loss, alpha, beta1, beta2, epsilon) which creates the training operation for a neural network in TensorFlow using the Adam optimization algorithm.

### 11-learning_rate_decay.py - Contains the function definition learning_rate_decay(alpha, decay_rate, global_step, decay_step) which updates the learning rate using inverse time decay.

### 12-learning_rate_decay.py - Contains the function definition learning_rate_decay(alpha, decay_rate, global_step, decay_step) which creates a learning rate decay operation in TensorFlow using inverse time decay.

### 13-batch_norm.py - Contains the function definition batch_norm(Z, gamma, beta, epsilon) which normalizes an unactivated output of a neural network (using batch normalization).

### 14-batch_norm.py - Contains the function definition create_batch_norm_layer(prev, n, activation) which creates a batch normalization layer for a neural network in TensorFlow.
