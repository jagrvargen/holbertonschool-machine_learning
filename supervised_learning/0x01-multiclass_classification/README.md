# Supervised Learning - Multiclass Classification

### 0-one_hot_encode.py - Contains a function which converts a numerically labeled numpy.ndarray (vector) to a one-hot encoded matrix.

### 1-one_hot_decode.py - Contains a function which converts a one-hot matrix into a vector of labels.

### 2-deep_neural_network.py - This file is an iteration on the class definition DeepNeuralNetwork from ../0x00-binary_classification/23-deep_neural_network.py which adds a method to save a model in pickle format as well as a method to load a model from pickle format.

### 3-deep_neural_network.py - Updates the DeepNeuralNetwork class method forward_prop to use the softmax activation function, the cost method to use cross-entropy loss, and the evaluate method to return a one-hot encoding of the network's prediction.

### 4-deep_neural_network.py -
- Updates the DeepNeuralNetwork class __init__ method with the parameter activation to allow for switching between sigmoid and tanh activations in the hidden layers of the network.
- A private instance attribute __activation is added to hold the value of the activation parameter upon instantiation of a DNN.
- The forward_prop and gradient_descent methods are updated to use the __activation attribute.
