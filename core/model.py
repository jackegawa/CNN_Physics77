import numpy as np
from .layers import Conv2D, LinearLayer    
from .operations import ReLUOP, FlattenOp    

class CNN:
    """
    A simple Convolutional Neural Network for MNIST classification.
    
    Architecture:
    1. Conv2D (1 -> out_channels, 3x3)
    2. ReLU
    3. Flatten
    4. Linear (Flat -> hidden_dim)
    5. ReLU
    6. Linear (hidden_dim -> 10 classes)
    """
    def __init__(self, out_channels=8, kernel_size=3, hidden_dim=128):
        # Architecture Hyperparameters
        self.in_channels = 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.num_classes = 10

        # Layer Definitions
        self.conv = Conv2D(self.in_channels, self.out_channels, self.kernel_size)
        self.relu = ReLUOP()
        self.flatten = FlattenOp()

        self.feature_h = 28 - self.kernel_size + 1
        self.feature_w = 28 - self.kernel_size + 1

        # Calculate flattened dimension for FC input
        self.flat_dim = self.out_channels * self.feature_h * self.feature_w
        
        self.fc1 = LinearLayer(self.flat_dim, self.hidden_dim)
        self.fc2 = LinearLayer(self.hidden_dim, self.num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def params(self):
        """Returns list of all trainable parameters."""
        p = []
        p += self.conv.params()
        p += self.fc1.params()
        p += self.fc2.params()
        return p
    
    def summary(self):
        """Returns a string list describing the model architecture."""
        return [
            f"Conv2D (In: {self.in_channels}, Out: {self.out_channels}, K: {self.kernel_size})",
            "ReLU",
            f"Flatten (Output: {self.flat_dim})",
            f"Linear (In: {self.flat_dim}, Out: {self.hidden_dim})",
            "ReLU",
            f"Linear (In: {self.hidden_dim}, Out: {self.num_classes})"
        ]