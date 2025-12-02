import numpy as np
from tensor import Tensor
"""
Here we define each operation type we will use,
This will also define the backward and forward pass for each operation

"""

class ConvOP:
    def __call__(self, x,weight):

        x_data = x.data  # (1, C, H, W)
        W = weight.data  # (K, C, 3, 3)

        batch, C, H, Wimg = x_data.shape
        K, Cw, Kh, Kw = W.shape

        out = np.zeros((batch, K, H - 2, Wimg - 2))

        """
        This is on the order of magnitude O(n^3), pretty bad. Need to vectorize and speed this up or we will be training forever.
        Same for backward pass
        """

        for k in range(K):
            for i in range(H - 2):
                for j in range(Wimg - 2):
                    patch = x_data[0, :, i:i+3, j:j+3]   # (C,3,3)
                    kernel = W[k]                        # (C,3,3)
                    out[0, k, i, j] = np.sum(patch * kernel)

        return Tensor(out, parents=[x, weight], op=self)

    def backward(self, out_tensor, grad):
        x, weight = out_tensor.parents
        x_data = x.data
        W_data = weight.data

        batch, C, H, Wimg = x_data.shape
        out_channels, Cw, Kh, Kw = W_data.shape
        out_h = H - 2
        out_w = Wimg - 2
        d_x = np.zeros_like(x_data)
        d_W = np.zeros_like(W_data)

        for k in range(out_channels):
            for i in range(H - 2):
                for j in range(Wimg - 2):
                    patch = x_data[0, :, i:i + 3, j:j + 3]  # (C,3,3)
                    g = grad[0, k, i, j]
                    d_W[k] += g * patch
                    d_x[0, :, i:i + 3, j:j + 3] += g * W_data[k]

        return [d_x, d_W]

class ReLUOP:
    def __call__(self, x):
        out = np.maximum(0,x.data)
        return Tensor(out, parents=[x], op=self)

    def backward(self,out, grad):
        x = out.parents[0]
        return [grad * (x.data > 0)]

# This is the same as overloading the @ operator and writing a backward pass for it
class MatMulOp:
    def __call__(self, x, w):
        return Tensor(x.data @ w.data, parents=[x, w], op=self)

    def backward(self,out, grad):
        x,w = out.parents
        d_x = grad @ w.data.T
        d_w = x.data.T @ grad
        return [d_x, d_w]


class AddOp:
    def __call__(self, x, w):
        return Tensor(x.data + w.data, parents=[x, w], op=self)
    def backward(self, out_tensor, d_out):
        x, b = out_tensor.parents
        return [d_out, np.sum(d_out, axis=0)]



class FlattenOp:
    def __call__(self, x):
        shape = x.data.shape
        batch = shape[0]
        out = x.data.reshape(batch,-1)
        tensor = Tensor(out, parents=[x], op=self)
        tensor._orig_shape = shape
        return tensor

    def backward(self, out_tensor, d_out):
        x = out_tensor.parents[0]
        return [d_out.reshape(out_tensor._orig_shape)]

class SoftMaxOp:
    def __call__(self, logits, y):

        self.labels = y
        #logits is a tensor so we take just the data
        x = logits.data
        #need to shift the logits to keep exponentials from exploding
        max_bit = np.max(x,axis=1,keepdims=True)
        z = np.exp(x - max_bit)
        soft = z/np.sum(z,axis=1,keepdims=True)

        self.softmax = soft
        # now we can compute the loss
        m = x.shape[0]
        log_likelihood = -np.log(soft[np.arange(m),self.labels])

        loss = np.mean(log_likelihood)

        return Tensor(loss, parents=[logits], op=self)



    def backward(self, out_tensor, grad):
        batch = self.softmax.shape[0]

        dx = self.softmax.copy()
        dx[np.arange(batch), self.labels] -= 1
        dx /=batch

        return [dx]


params =[]
