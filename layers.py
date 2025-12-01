import numpy as np
from tensor import Tensor
from operations import *

class Conv2D:
    def __init__(self,in_channels,out_channels,kernel_size):
        self.weights = Tensor(0.01 * np.random.randn(out_channels,in_channels,kernel_size,kernel_size))
        self.op = ConvOP()
    def __call__(self, x):
        return self.op(x,self.weights)
    def params(self):
        return [self.weights]

class LinearLayer:
    def __init__(self,in_features,out_features):
        self.weight = Tensor(0.01 * np.random.randn((in_features,out_features)))
        self.matmul = MatMulOp()
        self.add = AddOp()
        self.biases = np.zeros((in_features,out_features))

    def __call__(self,x):
        b = self.matmul(x,self.weight)
        return self.add(b,self.biases)

    def params(self):
        return [self.weight,self.biases]

