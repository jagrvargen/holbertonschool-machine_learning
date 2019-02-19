# Supervised Learning - Optimization

### 0-norm_constants.py - Contains the function definition normalization_constants(X) which calculates the normalization constants of a matrix.

### 1-normalize.py - Contains the function definition normalize(X, m, v, epsilon) which normalizes a matrix.

### 2-shuffle_data.py - Contains the function definition shuffle_data(X, Y) which shuffles the data points in two matrices in parallel.

### 3-mini_batch.py - Contains the function definition train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt") which trains a neural network using mini-batch gradient descent.

### 4-moving_average.py - Contains the function definition moving_average(data, beta) which calculates the weighted moving average of a data set.

### 5-momentum.py - Contains the function definition update_variables_momentum(alpha, beta1, var, grad, v) which updates a variable using the gradient descent with momentum optimization algorithm.

### 6-momentum.py - Contains the function definition create_momentum_op(loss, alpha, beta1) which creates a training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm.
