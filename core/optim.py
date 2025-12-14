import numpy as np

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Update rule:
    $$ \theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta) $$
    """
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            # Update weights in-place
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0)

    def summary(self):
        return {"optimizer": "SGD", "learning_rate": self.lr}

class Adam:
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Maintains moving averages of gradients ($m$) and squared gradients ($v$).
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Initialize buffers
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue

            g = p.grad

            # 1. Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # 2. Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            # 3. Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 4. Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0)

    def summary(self):
        return {
            "optimizer": "Adam",
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps
        }