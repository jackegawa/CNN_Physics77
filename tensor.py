import numpy as np

class Tensor:
    """
    A base Tensor class that stores data and gradients, 
    and supports automatic differentiation.
    """
    def __init__(self, data, parents=None, op=None):
        """
        Args:
            data (array-like): The numerical data (wrapped in np.float64).
            parents (list[Tensor], optional): Parent tensors in the computation graph.
            op (callable, optional): The operation that produced this tensor.
        """
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.parents = parents or []
        self.op = op

    def zero(self):
        """Resets gradients to zero."""
        self.grad = np.zeros_like(self.data)

    def backward(self, grad=None):
        """
        Computes gradients via backpropagation.
        Args:
            grad (np.ndarray, optional): Gradient flowing from the next layer. Defaluts to 1.0 if None for scalar outputs.
        """
        if grad is None:
            grad = np.ones_like(self.data)

        self.grad += grad

        if self.op:
            parent_grads = self.op.backward(self, grad)
            for parent, g in zip(self.parents, parent_grads):
                parent.backward(g)