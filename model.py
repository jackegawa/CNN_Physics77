import numpy as np
from layers import *
from operations import *


class CNN:
    def __init__(self):
        in_channels = 1
        out_channels = 8
        kernel_size = 3

        self.conv = Conv2D(in_channels, out_channels, kernel_size)

        self.relu = ReLUOP()
        self.flatten = FlattenOp()

        flat_dim = 8 * 26 * 26
        hidden_dim = 128
        num_classes = 10

        self.fc1 = LinearLayer(flat_dim, hidden_dim)
        self.fc2 = LinearLayer(hidden_dim, num_classes)


    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def params(self):
        p = []
        p += self.conv.params()
        p += self.fc1.params()
        p += self.fc2.params()
        return p