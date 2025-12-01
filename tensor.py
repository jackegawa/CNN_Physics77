import numpy as np

"""
Tensor object definition
Stores:
-Data
-Gradients
-Parent nodes
-Operation Type

"""

class Tensor:
    def __init__(self, data, parents=None, op=None):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.parents = parents or []
        self.op = op

    def zero(self):
        self.grad = np.zeros_like(self.data)

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        self.grad += grad

        if self.op:
            parent_grads = self.op.backward(self, grad)
            for parent, g in zip(self.parents, parent_grads):
                parent.backward(g)