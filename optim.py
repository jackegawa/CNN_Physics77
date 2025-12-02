import numpy as np

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Example Usage:
        # model.parameters() returns a list of Tensor objects
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # In training loop:
        # 1. zero_grad()
        # 2. loss.backward()
        # 3. optimizer.step()
    """
    def __init__(self, params, lr=0.01):
        """
        Args:
            params (list): A list of Tensor objects. Each object must have:
                           - .data (numpy array of weights)
                           - .grad (numpy array of gradients)
            lr (float): Learning rate.
        """
        self.params = list(params)  # Ensure it's a list
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        for p in self.params:
            # Skip parameters that don't have gradients (e.g. frozen layers)
            if p.grad is None:
                continue
            
            # Standard SGD update: theta = theta - lr * gradient
            # We assume 'p.data' holds the actual numpy array values
            p.data -= self.lr * p.grad

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        Must be called before the backward pass in the training loop.
        """
        for p in self.params:
            if p.grad is not None:
                # Reset gradient to zero (keeping the shape)
                p.grad.fill(0)