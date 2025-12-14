import numpy as np

class Tensor:
    """
    A base Tensor class that implements a node in the dynamic computation graph.
    
    It stores data, gradients, and the history of operations (parents) to support
    automatic differentiation (Autograd).

    Attributes:
        data (np.ndarray): The numerical value of the tensor.
        grad (np.ndarray): The gradient of the loss with respect to this tensor.
        parents (list[Tensor]): The input tensors that produced this tensor.
        op (callable): The operation that produced this tensor (for backward pass).
    """
    def __init__(self, data, parents=None, op=None):
        """
        Args:
            data (array-like): The numerical data (wrapped in np.float64).
            parents (list[Tensor], optional): Parent tensors in the computation graph.
            op (callable, optional): The operation class instance.
        """
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.parents = parents or []
        self.op = op

    def zero(self):
        """Resets gradients to zero. Crucial before accumulation in new steps."""
        self.grad = np.zeros_like(self.data)

    def backward(self, grad=None):
        """
        Computes gradients via backpropagation through the Directed Acyclic Graph (DAG).
        
        Logic:
            1. Accumulate incoming gradient into `self.grad`.
            2. If this tensor was produced by an operation, ask the operation 
               to calculate gradients for its parents (Chain Rule).
            3. Recursively call backward on parents.

        Args:
            grad (np.ndarray, optional): Gradient flowing from the next layer. 
                                         Defaults to 1.0 (scalar) for the loss node.
        """
        if grad is None:
            grad = np.ones_like(self.data)

        self.grad += grad

        if self.op:
            parent_grads = self.op.backward(self, grad)
            for parent, g in zip(self.parents, parent_grads):
                parent.backward(g)