# Supervised Learning - Convolutional Nerual Networks

### 0-conv_forward.py - Contains the function definition conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)) which performs forward propagation over a convolutional layer of a neural network.

### 1-pool_forward.py - Contains the function definition pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max') which performs forward propagation over a pooling layer of a neural network.

### 2-conv_backward.py - Contains the function definition conv_backward(dZ, A_prev, W, b, padding='same', stride=(1, 1)) which performs backward propagation over a convolutional layer of a neural network.

### 3-pool_backward.py - Contains the function definition pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max') which performs backpropagation over a pooling layer of a neural network.

### 4-lenet5.py - Contains the function definition lenet5(x, y) which builds a modified version of the LeNet-5 CNN architecture using TensorFlow.
