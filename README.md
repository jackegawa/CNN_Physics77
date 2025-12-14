# CNN_Physics77: NumPy-based Deep Learning Framework

This repository implements a Convolutional Neural Network (CNN) from scratch using **NumPy**, featuring a custom automatic differentiation engine (Autograd).

Designed as a computational physics project, this framework builds the deep learning stack from **first principles**. It includes a rigorous benchmarking suite that compares the custom implementation against a **PyTorch** baseline to validate mathematical correctness, convergence behavior, and memory efficiency.

## 🚀 Key Features

### 1. Core Implementations
+ **Custom Autograd Engine**: A dynamic computation graph with a `Tensor` class supporting gradient accumulation and automatic backpropagation (DAG).
+ **Vectorized Convolutions**: Implements `im2col` (**image-to-column**) and `col2im` algorithms to transform convolutions into efficient Matrix Multiplications (GEMM), leveraging NumPy's optimized BLAS routines.
+ **Modular Architecture**: Mimics PyTorch's API design with `Conv2D`, `Linear`, `ReLU`, and `Softmax` layers.
+ **Optimizers**: Custom implementations of **SGD** and **Adam** (with moment tracking and bias correction).

### 2. Scientific Benchmarking
An "Apples-to-Apples" comparison suite ensuring matched architecture, matched initialization scheme ($\mathcal{N}(0, 0.01)$), identical data, and controlled seeding for reproducibility to measure pure framework overhead:
+ **Efficiency Frontier**: Analyzing the trade-off between parameter count, accuracy, and training time.
+ **Gradient Stability**: Monitoring $||\nabla \theta||_2$ to ensure numerical stability in the custom backward pass.

## 📂 Project Structure

```bash
CNN_Physics77/
├── analysis/               # Visualization tools
│   └── visuals.py          # Plotting utilities (Loss, Accuracy, Efficiency Frontier)
├── benchmarks/             # Benchmarking scripts
│   ├── CNN_torch.py        # PyTorch Baseline (structurally aligned to custom model)
│   ├── train.py            # Custom Framework Training Loop
│   └── helper.py           # Experiment orchestration & logging
├── core/                   # The Custom Framework Library
│   ├── tensor.py           # Autograd & Tensor class
│   ├── operations.py       # Math Ops (im2col, Conv, ReLU, Softmax)
│   ├── layers.py           # Layer definitions (Conv2D, Linear)
│   ├── model.py            # CNN Architecture definition
│   ├── optim.py            # Optimizers (SGD, Adam)
│   └── data.py             # MNIST Data Loader
├── logs/                   # Training logs (JSON/TXT)
├── fig/                    # Generated plots
├── main.py                 # Entry point for running experiments
└── README.md
```

## 📦 Installation

1. **Environment Setup:**
    ```bash
    conda env create -f environment.yml
    conda activate cnn_physics77
    ```

2. **Clone the Repository:**
    ```bash
    git clone [https://github.com/jackegawa/CNN_Physics77](https://github.com/jackegawa/CNN_Physics77)
    cd CNN_Physics77
    ```

## 💻 Usage

The project is controlled via `main.py`, offering different modes for training and analysis.

### 1. Basic Benchmark (Custom vs PyTorch)
Runs a single comparison to verify that the Custom Model converges similarly to the PyTorch baseline.
```bash
python main.py --mode basic --epochs 5 --optimizer Adam --lr 0.001
```

### 2. Optimizer Grid Search
Runs a 4-way comparison (Custom-SGD, Custom-Adam, Torch-SGD, Torch-Adam) to analyze optimizer implementation correctness.
```bash
python main.py --mode grid --epochs 5 --lr 0.001
```

### 3. Diagnostics Mode
Generates detailed step-level analysis (Gradient Norms, Step Loss) to debug vanishing/exploding gradients.
```bash
python main.py --mode basic --diagnostics --epochs 5 --optimizer SGD --lr 0.001
```

### CLI Arguments

| Argument       | Default | Description                                                 |
|----------------|---------|-------------------------------------------------------------|
| `--mode`       | `basic` | Run mode: `basic` (1v1 comparison) or `grid` (4-way search) |
| `--optimizer`  | `Adam`  | Optimizer choice: `SGD` or `Adam`                           |
| `--epochs`     | `5`     | Number of training epochs                                   |
| `--batch_size` | `64`    | Batch size                                                  |
| `--lr`         | `0.001` | Learning rate                                               |
| `--seed`       | `67`    | Random seed for reproducibility                             |
| `--diagnostics`| `False` | Enable detailed diagnostics output                          |

## 📊 Visualizations

The `analysis/visuals.py` module automatically generates plots in the `fig/` directory. Key metrics include:

1. **Efficiency Frontier**: A bubble chart comparing **Model Size vs. Accuracy vs. Training Time**.

2. **Gradient Norm Stability**: Line plots of gradient norms over training steps to validate that the custom backward pass is mathematically stable.

3. **Loss & Accuracy Curves**: Standard training curves to confirm that both frameworks follow the same optimization trajectory.

## 🧠 Implementation Details

### The `im2col` Operation
Instead of using slow nested loops for convolution, this framework uses `im2col` to flatten input patches into a matrix. This converts the convolution operation into a single large Matrix Multiplication (GEMM):

$$
\text{Output} = \text{Conv2D}(\text{Input}, \text{Filters}) \implies \text{Output\_matrix} = \text{Input\_matrix} \times \text{Filter\_matrix}
$$

### Autograd Engine
The `Tensor` class maintains a list of `parents` and an `op` (operation). During the forward pass, the graph is built dynamically. During `backward()`, gradients flow using the chain rule:

```python
# Simplified logic from core/tensor.py
def backward(self, grad):
    self.grad += grad
    if self.op:
        parent_grads = self.op.backward(self, grad)
        for parent, g in zip(self.parents, parent_grads):
            parent.backward(g)
```

> **Note on Performance:** As this framework is implemented purely in Python/NumPy for educational purposes, it is expected to be slower than PyTorch (which relies on C++/CUDA backends), especially for larger batch sizes.