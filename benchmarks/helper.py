from dataclasses import dataclass
import tracemalloc
import json
import time
import random
import numpy as np
import os

from analysis import visuals

# ============================================================
# Reproducibility Utilities
# ============================================================

def set_seed(seed: int):
    """
    Set random seed for Python, NumPy, and PyTorch (if available)
    to ensure reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================
# Memory Utilities
# ============================================================

def get_memory_snapshot():
    """Returns (Current MB, Peak MB)."""
    current, peak = tracemalloc.get_traced_memory()
    return current / 1e6, peak / 1e6


# ============================================================
# Logging Utilities
# ============================================================

def log_to_json(
    filename: str,
    param_count: int,
    lr: float,
    batch_size: int,
    num_epochs: int,
    log_data: dict,
    model,
    optimizer_type: str,
    optimizer
):
    if hasattr(optimizer, 'summary'):
        optimizer_summary = optimizer.summary()
    elif isinstance(optimizer, dict):
        optimizer_summary = optimizer
    else:
        optimizer_summary = str(optimizer)

    if hasattr(model, 'summary'):
        model_architecture = model.summary()
    else:
        model_architecture = str(model)

    final_log = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "config": {
            "model_name": model.__class__.__name__,
            "parameter_count": param_count,
            "architecture": model_architecture,
            "optimizer_type": optimizer_type,
            "optimizer": optimizer_summary,
            "hyperparameters": {
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": lr
            }
        },
        "training_results": log_data
    }

    with open(filename, 'w') as f:
        json.dump(final_log, f, indent=4)

    print(f"\n[INFO] Log saved successfully to: {filename}")
    return filename


# ============================================================
# Experiment Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    # Optimizer & training hyperparameters
    optimizer_type: str = "Adam"     # "Adam" or "SGD"
    num_epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 67

    # CNN architecture
    model_out_channels: int = 8
    model_kernel_size: int = 3
    model_hidden_dim: int = 128

    # Logging
    log_name: str = "benchmark_run"
    log_path: str = "./logs"


# ============================================================
# Training Runners
# ============================================================

def run_custom(cfg: ExperimentConfig) -> str:
    """Run training in the custom NumPy CNN framework."""
    # Absolute import to allow running from root via main.py
    from benchmarks import train as custom_model

    print(f">>> Running Custom framework with {cfg.optimizer_type} ...")
    set_seed(cfg.seed)

    log_file = custom_model.train(
        optimizer_type=cfg.optimizer_type,
        log_name=f"CNN_{cfg.log_name}",
        log_path=cfg.log_path,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        seed=cfg.seed,
        model_out_channels=cfg.model_out_channels,
        model_kernel_size=cfg.model_kernel_size,
        model_hidden_dim=cfg.model_hidden_dim,
    )

    print("Custom log saved at:", log_file)
    return log_file


def run_torch(cfg: ExperimentConfig) -> str:
    """Run training in the PyTorch baseline."""
    # Absolute import
    from benchmarks import CNN_torch as torch_model

    print(f">>> Running PyTorch baseline with {cfg.optimizer_type} ...")
    set_seed(cfg.seed)

    log_file = torch_model.train_torch(
        optimizer_type=cfg.optimizer_type,
        log_name=f"TorchCNN_{cfg.log_name}",
        log_path=cfg.log_path,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        seed=cfg.seed,
        model_out_channels=cfg.model_out_channels,
        model_kernel_size=cfg.model_kernel_size,
        model_hidden_dim=cfg.model_hidden_dim,
    )

    print("Torch log saved at:", log_file)
    return log_file


def run_pair(cfg: ExperimentConfig):
    custom_log = run_custom(cfg)
    torch_log = run_torch(cfg)
    return custom_log, torch_log


# ============================================================
# Grid Search
# ============================================================

def run_optimizer_grid(base_cfg: ExperimentConfig):
    logs = {}

    # 1. SGD Config
    cfg_sgd = ExperimentConfig(
        optimizer_type="SGD",
        num_epochs=base_cfg.num_epochs,
        batch_size=base_cfg.batch_size,
        lr=base_cfg.lr,
        seed=base_cfg.seed,
        model_out_channels=base_cfg.model_out_channels,
        model_kernel_size=base_cfg.model_kernel_size,
        model_hidden_dim=base_cfg.model_hidden_dim,
        log_name=base_cfg.log_name,
        log_path=base_cfg.log_path,
    )

    # 2. Adam Config
    cfg_adam = ExperimentConfig(
        optimizer_type="Adam",
        num_epochs=base_cfg.num_epochs,
        batch_size=base_cfg.batch_size,
        lr=base_cfg.lr,
        seed=base_cfg.seed,
        model_out_channels=base_cfg.model_out_channels,
        model_kernel_size=base_cfg.model_kernel_size,
        model_hidden_dim=base_cfg.model_hidden_dim,
        log_name=base_cfg.log_name,
        log_path=base_cfg.log_path,
    )

    logs["custom_sgd"] = run_custom(cfg_sgd)
    logs["custom_adam"] = run_custom(cfg_adam)
    logs["torch_sgd"] = run_torch(cfg_sgd)
    logs["torch_adam"] = run_torch(cfg_adam)

    return logs


# ============================================================
# Visualization Wrappers
# ============================================================

def visualize_pair(custom_log: str, torch_log: str):
    print("\n[PLOT] Custom vs PyTorch: Convergence & Generalization")
    visuals.plot_loss_comparison(custom_log, torch_log)
    visuals.plot_accuracy_comparison(custom_log, torch_log)

    print("\n[PLOT] Training Stability: Gradient Norms")
    # Fix: Added explicit titles to prevent filename collision
    visuals.plot_gradient_norm(custom_log, window=50, title="Custom Framework Gradient Norms")
    visuals.plot_gradient_norm(torch_log, window=50, title="PyTorch Baseline Gradient Norms")

    print("\n[PLOT] Efficiency Frontier: Time vs Memory")
    visuals.plot_efficiency_frontier([custom_log, torch_log])

    print("\n[PLOT] System Benchmark (Param count, time, memory)")
    visuals.plot_system_benchmark(custom_log, torch_log)


def visualize_diagnostics(log_path: str, title_prefix: str = ""):
    print(f"\n[PLOT] Detailed Training Dynamics for {title_prefix}")
    visuals.plot_detailed_loss(
        log_path, window=100,
        title=f"{title_prefix}: Detailed Step Loss"
    )

    print(f"\n[PLOT] Gradient Norm Stability for {title_prefix}")
    visuals.plot_gradient_norm(
        log_path, window=50,
        title=f"{title_prefix}: Gradient Norms"
    )

    print(f"\n[PLOT] Confusion Matrix for {title_prefix}")
    visuals.plot_confusion_matrix(log_path)


def visualize_optimizer_grid(logs: dict):
    print("\n[PLOT] Generating 4-Set Comparative Visualization (Optimizers x Frameworks)...")

    all_logs = [
        logs["custom_sgd"],
        logs["custom_adam"],
        logs["torch_sgd"],
        logs["torch_adam"],
    ]

    visuals.plot_multi_loss(
        all_logs,
        title="Training Loss Comparison Across Optimizers"
    )

    visuals.plot_multi_accuracy(
        all_logs,
        title="Test Accuracy Comparison Across Optimizers"
    )

    visuals.plot_multi_grad_norm(
        all_logs,
        window=50,
        title="Gradient Norm Comparison Across Optimizers"
    )

    visuals.plot_performance_bar(
        all_logs,
        title="Training Speed & Memory Comparison Across Optimizers"
    )