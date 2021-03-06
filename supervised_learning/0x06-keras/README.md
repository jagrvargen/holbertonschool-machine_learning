# Supervised Learning - Keras

### 0-sequential.py - Contains the function definition build_model(nx, layers, activations, lambtha, keep_prob) which builds a neural network with the Keras library.

### 1-input.py - Contains the function definition build_model(nx, layers, activations, lambtha, keep_prob) which builds a neural network with the Keras library.

### 2-compile.py - Contains the function definition optimize_model(network, alpha, beta1, beta2) which compiles the model with Adam optimization.

### 3-one_hot.py - Contains the function definition one_hot(labels, classes=None) which converts a label vector into a one hot matrix.

### 4-train.py - Contains the function definition train_model(network, data, labels, batch_size, epochs, verbose=True) which trains a model using gradient descent.

### 5-train.py - Contains the function definition train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True) which trains a model using mini-batch gradient descent.

### 6-train.py - Contains the function definition train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True) which trains a model using mini-batch gradient descent.

### 7-train.py - Contains the function definition train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True) which trains a model using mini-batch gradient descent.

### 8-train.py - Contains the function definition train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True) which trains a model using mini-batch gradient descent.

### 9-model.py - Contains the function definitions save_model(network, filename) and load_model(filename) which save and load a Keras model respectively.

### 10-weights.py - Contains the function definitions save_weights(network, filename, save_format='h5') and load_weights(network, filename) which save and load a Keras model's weights respectively.

### 11-config.py - Contains the function definitions save_config(network, filename) and load_config(filename) which save and load a Keras model's configuration in JSON format respectively.

### 12-test.py - Contains the function definition test_model(network, data, labels, verbose=True) which tests a neural network.

### 13-predict.py - Contains the function definition predict(network, data, verbose=False) which makes a prediction using an existing neural network.
