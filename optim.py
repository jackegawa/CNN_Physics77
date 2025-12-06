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
    def __init__(self, params, lr=0.001):
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

    def schedule_lr(self, gamma):
        """
        Decays the learning rate by a factor of gamma.
        
        Args:
            gamma (float): Factor to multiply the learning rate by.
        """
        self.lr *= gamma

    def summary(self):
        """
        Returns a summary of the optimizer configuration.
        """
        return {
            "optimizer": "SGD",
            "learning_rate": self.lr
        }

class Adam:
    """
    Adam optimizer (NumPy version).
    """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # State buffers (same shape as parameters)
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

        self.t = 0  # time step

    def step(self):
        self.t += 1  # increase timestep

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            # Compute bias-corrected moments
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0)

    def schedule_lr(self, gamma):
        """ decay learning rate """
        self.lr *= gamma

    def summary(self):
        """
        Returns a summary of the optimizer configuration.
        """
        return {
            "optimizer": "Adam",
            "learning_rate": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.eps
        }