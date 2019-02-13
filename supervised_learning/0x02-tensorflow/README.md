# Supervised Learning - TensorFlow

### 0-create_placeholders.py - Defines the function create_placeholders(nx, classes) which returns two TensorFlow placeholders x and y.

### 1-create_layer.py - Defines the function create_layer(prev, n_prev, n, activation) which creates a layer for a neural network in TensorFlow.

### 2-forward_prop.py - Defines the function forward_prop(x, nx, layer_sizes=[], activations=[]) which creates the forward propagation graph for a neural network.

### 3-calculate_accuracy.py - Defines the function calculate_accuracy(y, y_pred) which calculates the accuracy of the prediction made by a neural network.

### 4-calculate_loss.py - Defines the function calculate_loss(y, y_pred) which calculates the softmax cross-entropy loss of a neural network's prediction.

### 5-create_train_op.py - Defines the function create_train_op(loss, alpha) which creates the training operation for a neural network.

### 6-train.py - Defines the function train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt") which builds, trains, and saves a neural network classifier.
