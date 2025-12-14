import numpy as np
from .tensor import Tensor  
from .operations import * 

class Conv2D:
    """
    Convolutional Layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        # Weight shape: (Out, In, k, k)
        self.weights = Tensor(0.01 * np.random.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.op = ConvOP()

    def __call__(self, x):
        return self.op(x, self.weights)

    def params(self):
        return [self.weights]

class LinearLayer:
    """
    Fully Connected (Dense) Layer.
    """
    def __init__(self, in_features, out_features):
        self.weight = Tensor(0.01 * np.random.randn(in_features, out_features))
        self.biases = Tensor(np.zeros((1, out_features)))
        
        self.matmul = MatMulOp()
        self.add = AddOp()

    def __call__(self, x):
        # Y = XW + b
        b = self.matmul(x, self.weight)
        return self.add(b, self.biases)

    def params(self):
        return [self.weight, self.biases]
