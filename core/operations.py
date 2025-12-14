import numpy as np
from .tensor import Tensor

"""
Mathematical Operations & Autograd Functions
============================================

This module defines the forward and backward passes for CNN building blocks.
All operations must support the autograd engine by saving parent tensors
and implementing a `backward` method returning gradients for inputs.
"""

def im2col(input_data, kernel_size):
    """
    Vectorizes image blocks for convolution using stride tricks.
    
    Transforms the 4D input tensor into a 2D matrix where each column 
    corresponds to a kernel-sized patch of the image.

    Args:
        input_data (np.ndarray): Shape (B, C, H, W)
        kernel_size (int): Size of the square kernel (k)

    Returns:
        patches (np.ndarray): Shape (B, OH*OW, C*k*k)
    """
    B, C, H, W = input_data.shape
    kH = kW = kernel_size
    out_h = H - kH + 1
    out_w = W - kW + 1
    
    # Create a strided view of the array (virtual copies)
    shape = (B, out_h, out_w, C, kH, kW)
    strides = (
        input_data.strides[0],
        input_data.strides[2],
        input_data.strides[3],
        input_data.strides[1],
        input_data.strides[2],
        input_data.strides[3],
    )

    patches = np.lib.stride_tricks.as_strided(input_data, shape=shape, strides=strides)
    patches = patches.reshape(B, out_h * out_w, C * kH * kW)
    return patches

def col2im(cols, in_shape, kernel_size):
    """
    Inverse of im2col, used during the backward pass to reconstruct gradients.
    
    Args:
        cols (np.ndarray): Flattened gradients from GEMM.
        in_shape (tuple): Original shape (B, C, H, W).
        kernel_size (int): Kernel size k.

    Returns:
        dx (np.ndarray): Reconstructed gradient tensor (B, C, H, W).
    """
    B, C, H, W = in_shape
    kH = kW = kernel_size
    OH = H - kH + 1
    OW = W - kW + 1

    cols = cols.reshape(B, OH, OW, C, kH, kW)
    # Permute to (B, C, OH, OW, kH, kW)
    cols = cols.transpose(0, 3, 1, 2, 4, 5)

    dx = np.zeros((B, C, H, W))

    # Overlap-add to accumulate gradients where patches overlapped
    for i in range(kH):
        for j in range(kW):
            dx[:, :, i:i + OH, j:j + OW] += cols[:, :, :, :, i, j]

    return dx


class ConvOP:
    """
    Forward: $$Y = X * W$$ via GEMM (im2col).
    Backward: Computes $dL/dX$ and $dL/dW$.
    """
    def __call__(self, x, weight):
        """
        Args:
            x (Tensor): Input (B, C, H, W)
            weight (Tensor): Filters (K, C, k, k)
        """
        x_data = x.data  
        W = weight.data 

        B, C, H, Wimg = x_data.shape
        K, Cw, kH, kW = W.shape
        OH = H - kH + 1
        OW = Wimg - kW + 1

        # 1. Vectorize input: (B, OH*OW, C*k*k)
        patches = im2col(x_data, kH) 

        # 2. Reshape weights: (C*k*k, K)
        W_col = W.reshape(K, -1).T    

        # 3. GEMM: (B, OH*OW, K)
        out = patches @ W_col         

        # 4. Reshape output: (B, K, OH, OW)
        out = out.reshape(B, OH, OW, K).transpose(0, 3, 1, 2)

        return Tensor(out, parents=[x, weight], op=self)

    def backward(self, out_tensor, grad):
        x, weight = out_tensor.parents
        x_data = x.data
        W = weight.data

        B, C, H, Wimg = x_data.shape
        K, Cw, kH, kW = W.shape
        OH = H - kH + 1
        OW = Wimg - kW + 1

        # Reshape grad: (B, K, OH, OW) -> (B, OH*OW, K)
        grad_col = grad.transpose(0, 2, 3, 1).reshape(B, OH*OW, K)

        # 1. dL/dW: (grad)^T @ patches
        patches = im2col(x_data, kH)
        dW = np.zeros_like(W)

        # Sum gradients over the batch dimension
        for b in range(B):
            dW += (grad_col[b].T @ patches[b]).reshape(W.shape)

        # 2. dL/dX: grad @ W^T -> col2im
        W_col = W.reshape(K, -1).T
        dx_col = grad_col @ W_col.T

        dx = col2im(dx_col, x_data.shape, kH)

        return [dx, dW]


class ReLUOP:
    """Rectified Linear Unit: $$f(x) = \max(0, x)$$"""
    def __call__(self, x):
        out = np.maximum(0, x.data)
        return Tensor(out, parents=[x], op=self)

    def backward(self, out, grad):
        x = out.parents[0]
        # Pass gradient only where input > 0
        return [grad * (x.data > 0)]

class MatMulOp:
    """Matrix Multiplication: $$Y = X W^T$$"""
    def __call__(self, x, w):
        # x: (B, in), w: (out, in) -> out: (B, out) via x @ w.T
        return Tensor(x.data @ w.data, parents=[x, w], op=self)

    def backward(self, out, grad):
        x, w = out.parents
        d_x = grad @ w.data.T
        d_w = x.data.T @ grad
        return [d_x, d_w]


class AddOp:
    """Bias Addition with Broadcasting."""
    def __call__(self, x, w):
        return Tensor(x.data + w.data, parents=[x, w], op=self)
        
    def backward(self, out_tensor, d_out):
        x, b = out_tensor.parents
        # Sum over batch dimension for the bias gradient
        return [d_out, np.sum(d_out, axis=0)]


class FlattenOp:
    """Reshapes (B, C, H, W) -> (B, Features)."""
    def __call__(self, x):
        shape = x.data.shape
        batch = shape[0]
        out = x.data.reshape(batch, -1)
        tensor = Tensor(out, parents=[x], op=self)
        tensor._orig_shape = shape # Save shape for backward
        return tensor

    def backward(self, out_tensor, d_out):
        return [d_out.reshape(out_tensor._orig_shape)]

class SoftMaxOp:
    """
    Combined Softmax + Cross Entropy Loss.
    
    $$Loss = - \log \left( \frac{e^{x_y}}{\sum e^{x_j}} \right)$$
    """
    def __call__(self, logits, y):
        self.labels = y
        x = logits.data
        
        # Numerical stability shift: exp(x - max)
        max_bit = np.max(x, axis=1, keepdims=True)
        z = np.exp(x - max_bit)
        soft = z / np.sum(z, axis=1, keepdims=True)

        self.softmax = soft
        
        # Calculate Cross Entropy Loss
        m = x.shape[0]
        log_likelihood = -np.log(soft[np.arange(m), self.labels])
        loss = np.mean(log_likelihood)

        return Tensor(loss, parents=[logits], op=self)

    def backward(self, out_tensor, grad):
        """
        Gradient of CE + Softmax is simply: $$p - y$$
        """
        batch = self.softmax.shape[0]
        dx = self.softmax.copy()
        
        # Subtract 1 from the correct class probability
        dx[np.arange(batch), self.labels] -= 1
        
        # Average over batch
        dx /= batch

        return [dx]