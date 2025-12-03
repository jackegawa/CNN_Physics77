
import numpy as np
from tensor import Tensor
"""
Here we define each operation type we will use,
This will also define the backward and forward pass for each operation

"""

def im2col(input,kernel_size):
    """
    Helper function to vectorize our conv op
    Turns an image into a column vector
    """
    B,C,H,W = input.shape
    kH = kW = kernel_size
    out_h = H - kH + 1
    out_w = W - kW + 1
    shape = (B, out_h, out_w, C, kH, kW)
    strides = (
        input.strides[0],
        input.strides[2],
        input.strides[3],
        input.strides[1],
        input.strides[2],
        input.strides[3],
    )

    patches = np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)
    patches = patches.reshape(B, out_h * out_w, C * kH * kW)
    return patches

def col2im(cols,in_shape, kernel_size):
    B, C, H, W = in_shape
    kH = kW = kernel_size
    OH = H - kH + 1
    OW = W - kW + 1
    cols = cols.reshape(B, OH, OW, C, kH, kW)
    cols = cols.transpose(0, 3, 1, 2, 4, 5)
    dx = np.zeros((B, C, H, W))

    for i in range(kH):
        for j in range(kW):
            dx[:, :, i:i + OH, j:j + OW] += cols[:, :, :, :, i, j]

    return dx


class ConvOP:
    def __call__(self, x,weight):

        x_data = x.data  # (1, C, H, W)
        W = weight.data  # (K, C, 3, 3)

        B, C, H, Wimg = x_data.shape
        K, Cw, kH, kW = W.shape
        OH = H - kH + 1
        OW = Wimg - kW + 1

        patches = im2col(x_data, 3) #get patches

        W_col = W.reshape(K, -1).T #shape weights

        out = patches @ W_col  # (B, OH*OW, K)
        out = out.reshape(B, OH, OW, K).transpose(0, 3, 1, 2)

        return Tensor(out, parents=[x, weight], op=self)

    def backward(self, out_tensor, grad):
        x,weight = out_tensor.parents
        x_data = x.data
        W = weight.data

        B, C, H, Wimg = x_data.shape
        K, Cw, kH, kW = W.shape

        OH = H - kH + 1
        OW = Wimg - kW + 1

        #shape grad
        grad_col = grad.transpose(0, 2, 3, 1).reshape(B, OH*OW, K)

        patches = im2col(x_data, kH)
        dW = np.zeros_like(W)

        for b in range(B):
            dW += (grad_col[b].T @ patches[b]).reshape(W.shape)

        W_col = W.reshape(K, -1).T
        dx_col = grad_col @ W_col.T


        dx = col2im(dx_col,x_data.shape, kH)

        return [dx, dW]


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
