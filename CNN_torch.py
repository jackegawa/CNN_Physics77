import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data import load_mnist
import numpy as np
import time
import tracemalloc
import json
import os
import random
import sys
from contextlib import redirect_stdout
from helper import log_to_json, get_memory_snapshot

# ==========================================
# 1. Model Definition (Aligned with Custom)
# ==========================================
class TorchCNN(nn.Module):
    """
    A PyTorch implementation of the custom CNN defined in 'model.py'.
    
    This model is structurally aligned with the custom NumPy implementation
    to ensure fair benchmarking. It explicitly sets bias terms and 
    initialization methods to match the custom framework.
    """
    def __init__(self, out_channels=8, kernel_size=3, hidden_dim=128):
        """
        Initializes the CNN architecture.

        Args:
            out_channels (int): Number of output channels for the convolutional layer.
            kernel_size (int): Size of the convolving kernel.
            hidden_dim (int): Number of units in the hidden fully connected layer.
        """
        super(TorchCNN, self).__init__()
        
        # Save architecture parameters for logging purposes
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

        # [ALIGNMENT] The custom Conv2D in layers.py does NOT implement bias.
        # Therefore, we must set bias=False here to match parameters exactly.
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=False 
        )

        self.relu = nn.ReLU()
        
        # Calculate Flatten Dimension dynamically based on kernel size
        # Input image: 28x28 (MNIST)
        self.feature_h = 28 - kernel_size + 1
        self.feature_w = 28 - kernel_size + 1
        self.flat_dim = out_channels * self.feature_h * self.feature_w

        # [ALIGNMENT] The custom LinearLayer in layers.py HAS bias.
        # So we keep bias=True (default) for linear layers.
        self.fc1 = nn.Linear(self.flat_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 10, bias=True)

        self._init_weights_custom()

    def _init_weights_custom(self):
        """
        Custom weight initialization to match 'layers.py'.
        
        Logic:
            - Custom framework uses: Tensor(0.01 * np.random.randn(...))
            - This corresponds to a Normal distribution with mean=0.0 and std=0.01.
            - Biases are initialized to 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 1, 28, 28).

        Returns:
            torch.Tensor: Logits of shape (Batch, 10).
        """
        out = self.conv(x)
        out = self.relu(out)
        
        # Flatten the output for the fully connected layers
        out = out.view(-1, self.flat_dim)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def summary(self):
        """
        Returns a list describing the model architecture.
        
        Returns:
            list[str]: A list of strings describing layers, matching the format 
                       used in the custom model's summary.
        """
        return [
            f"Conv2d(1 -> {self.out_channels}, k={self.kernel_size}, bias=False)",
            "ReLU",
            f"Flatten({self.flat_dim})",
            f"Linear({self.flat_dim} -> {self.hidden_dim})",
            "ReLU",
            f"Linear({self.hidden_dim} -> 10)"
        ]

# ==========================================
# 2. Helper Functions
# ==========================================

def get_data_loaders(batch_size=64):
    """
    Creates DataLoaders for the MNIST dataset using the SAME numpy arrays
    as our custom framework (data.load_mnist).
    """
    # Use the same data loading function to ensure identical data
    x_train, y_train, x_test, y_test = load_mnist()

    # Convert numpy arrays to PyTorch tensors
    x_train_t = torch.from_numpy(x_train).float()   # (N, 1, 28, 28), [0,1]
    x_test_t  = torch.from_numpy(x_test).float()
    y_train_t = torch.from_numpy(y_train).long()    # labels
    y_test_t  = torch.from_numpy(y_test).long()

    # Create TensorDatasets
    train_dataset = TensorDataset(x_train_t, y_train_t)
    test_dataset  = TensorDataset(x_test_t,  y_test_t)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    test_loader  = DataLoader(test_dataset,
                              batch_size=1000,
                              shuffle=False,
                              num_workers=0)

    return train_loader, test_loader

def get_optimizer_summary(optimizer, type_str):
    """
    Extracts configuration details from a PyTorch optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.
        type_str (str): The name of the optimizer ("Adam" or "SGD").

    Returns:
        dict: A dictionary containing optimizer hyperparameters.
    """
    defaults = optimizer.defaults
    if type_str == "Adam":
        return {
            "type": "Adam",
            "lr": defaults['lr'],
            "beta1": defaults['betas'][0],
            "beta2": defaults['betas'][1],
            "epsilon": defaults['eps']
        }
    else:
        return {
            "type": "SGD",
            "lr": defaults['lr']
        }

# ==========================================
# 3. Main Training Function (Matches train.py)
# ==========================================

def train_torch(
    optimizer_type="Adam",
    log_name="torch_benchmark",
    log_path="./logs",
    num_epochs=5,
    batch_size=64,
    lr=0.001,
    model_out_channels=8,
    model_kernel_size=3,
    model_hidden_dim=128,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
):
    """
    Main training loop for the PyTorch implementation.
    
    This function mirrors the logic of the custom `train.py` script, including
    memory tracking, gradient norm logging, and JSON artifact generation.

    Args:
        optimizer_type (str): "Adam" or "SGD".
        log_name (str): Base name for the output log file.
        log_path (str): Directory to save logs.
        num_epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.
        lr (float): Learning rate.
        model_out_channels (int): Number of filters in Conv layer.
        model_kernel_size (int): Size of Conv kernel.
        model_hidden_dim (int): Size of hidden FC layer.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        epsilon (float): Adam epsilon.

    Returns:
        str: The full path to the saved JSON log file.
    """
    
    # Default to CPU for a fair comparison with the NumPy-only custom framework.
    # Change to 'cuda' or 'mps' only if you want to benchmark hardware acceleration.
    # RNG Seed
    # RNG Seed
    seed = 67
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cpu") 

    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"{log_name}_k{model_kernel_size}_c{model_out_channels}_h{model_hidden_dim}_{optimizer_type}_lr{lr}_{timestamp_str}"
    log_filename = f"{log_path}/{base_filename}.json"
    txt_filename = f"{log_path}/{base_filename}.txt"
    
    print(f"[INFO-TORCH] Training Start | Opt: {optimizer_type} | Device: {device}")
    with open(txt_filename, 'w') as f_log:
        with redirect_stdout(f_log):
            # Ignore warnings on MacOS
            import warnings
            warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
            warnings.filterwarnings("ignore", message="overflow encountered in matmul")
            warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

            # --- Initialization ---
            tracemalloc.start()

            # 1. Load Data
            print(f"[INFO] Loading MNIST dataset...", end=" ")
            train_loader, test_loader = get_data_loaders(batch_size)
            print(f"Done. (Train Batches: {len(train_loader)})")

            current, peak = get_memory_snapshot()
            print(f"[INFO] Initial Memory: {current:.2f} MB | Peak Memory: {peak:.2f} MB")

            # 2. Build Model
            model = TorchCNN(
                out_channels=model_out_channels,
                kernel_size=model_kernel_size,
                hidden_dim=model_hidden_dim
            ).to(device)

            # Calculate Parameter Count
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[INFO] Model initialized with {param_count:,} parameters.")

            # 3. Optimizer
            if optimizer_type == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
            elif optimizer_type == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

            criterion = nn.CrossEntropyLoss()

            # 4. Prepare Logging Structure (Exact match with train.py)
            log_data = {
                "per_epoch_stats": [],
                "detailed_loss": [],
                "detailed_grad_norm": [],
                "final_evaluation": {}
            }

            # --- Training Loop ---
            total_start = time.time()

            for epoch in range(num_epochs):
                epoch_start = time.time()
                print(f"\n[TRAIN] EPOCH {epoch+1}/{num_epochs}")
                print(f"-"*60)
                
                model.train()
                running_loss = 0.0
                processed_samples = 0
                epoch_grad_norms = []
                
                # Batch Loop
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    current_batch_size = data.size(0)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()

                    # [ALIGNMENT] Calculate Gradient Norm before optimizer step
                    # This mirrors the manual calculation in the custom framework.
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    epoch_grad_norms.append(total_norm)
                    log_data["detailed_grad_norm"].append(float(total_norm))

                    optimizer.step()

                    # Logging
                    loss_val = loss.item()
                    running_loss += loss_val * current_batch_size
                    processed_samples += current_batch_size
                    log_data["detailed_loss"].append(loss_val)

                    # Console Log
                    if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                        current, peak = get_memory_snapshot()
                        progress = (batch_idx + 1) / len(train_loader) * 100
                        print(f"   Step {batch_idx+1:3d}/{len(train_loader)} ({progress:5.1f}%) | "
                            f"Loss: {loss_val:.4f} | Grad: {total_norm:.2f} | Mem: {current:.1f} MB")

                # Epoch Summary
                avg_loss = running_loss / processed_samples
                avg_grad = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
                epoch_time = time.time() - epoch_start

                # --- Evaluation ---
                print(f"[EVAL]  Validating...", end="\r")
                model.eval()
                correct = 0
                total_test_samples = 0
                
                collect_preds = (epoch == num_epochs - 1)
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        pred = output.argmax(dim=1)
                        
                        correct += (pred == target).sum().item()
                        total_test_samples += target.size(0)

                        if collect_preds:
                            all_preds.extend(pred.cpu().numpy().tolist())
                            all_labels.extend(target.cpu().numpy().tolist())

                acc = correct / total_test_samples
                _, final_peak = get_memory_snapshot()

                print(f"[RESULT] Epoch {epoch+1} Finished in {epoch_time:.2f}s | "
                    f"Train Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}% | Peak Mem: {final_peak:.1f} MB")

                log_data["per_epoch_stats"].append({
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "accuracy": acc,
                    "time": epoch_time,
                    "memory_peak": final_peak,
                    "avg_grad_norm": float(avg_grad)
                })

                if collect_preds:
                    log_data["final_evaluation"] = {
                        "y_true": all_labels,
                        "y_pred": all_preds
                    }

            # --- Finalize & Save JSON ---
            print(f"\n{'='*60}")
            print(f"[INFO] All training completed in {time.time() - total_start:.2f}s")

            log_file = log_to_json(
                filename=log_filename,
                param_count=param_count,
                lr=lr,
                batch_size=batch_size,
                num_epochs=num_epochs,
                log_data=log_data,
                model=model,
                optimizer_type=optimizer_type,
                optimizer=get_optimizer_summary(optimizer, optimizer_type)
            )
            
            tracemalloc.stop()
    print(f"Training log saved to: {log_file}")
    return log_file