from dataclasses import dataclass
import tracemalloc
import json
import time

import visuals

def get_memory_snapshot():
    """Returns (Current MB, Peak MB)"""
    current, peak = tracemalloc.get_traced_memory()
    return current / 10**6, peak / 10**6

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
    """
    Saves experiment artifacts to a JSON file.
    Args:
        filename (str): Path to save the log file.
        param_count (int): Number of model parameters.
        lr (float): Learning rate used during training.
        batch_size (int): Training batch size.
        num_epochs (int): Number of training epochs.
        log_data (dict): Collected log data during training.
        model: The trained model instance.
        optimizer_type (str): Type of optimizer used during training.
        optimizer: The optimizer instance used during training.
    Returns:
        dict: The final structured log data.
    """
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
    # Clean structured log
    final_log = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "config": {
            "model_name": model.__class__.__name__,
            "parameter_count": param_count,
            "architecture": model.summary(),
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

@dataclass
class ExperimentConfig:
    # Optimizer & training hyperparameters
    optimizer_type: str = "Adam"     # "Adam" or "SGD"
    num_epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3

    # CNN architecture
    model_out_channels: int = 8
    model_kernel_size: int = 3
    model_hidden_dim: int = 128

    # Logging
    log_name: str = "benchmark_run"
    log_path: str = "./logs"

def run_custom(cfg: ExperimentConfig) -> str:
    """
    Run training in our Custom NumPy framework and return the JSON log path.
    """
    import train as custom_model
    print(f">>> Running Custom framework with {cfg.optimizer_type} ...")
    log_file = custom_model.train(
        optimizer_type=cfg.optimizer_type,
        log_name=f"CNN_{cfg.log_name}",
        log_path=cfg.log_path,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        model_out_channels=cfg.model_out_channels,
        model_kernel_size=cfg.model_kernel_size,
        model_hidden_dim=cfg.model_hidden_dim,
    )
    print("Custom log saved at:", log_file)
    return log_file


def run_torch(cfg: ExperimentConfig) -> str:
    """
    Run training in the PyTorch baseline and return the JSON log path.
    """
    import CNN_torch as torch_model
    print(f">>> Running PyTorch baseline with {cfg.optimizer_type} ...")
    log_file = torch_model.train_torch(
        optimizer_type=cfg.optimizer_type,
        log_name=f"TorchCNN_{cfg.log_name}",
        log_path=cfg.log_path,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        model_out_channels=cfg.model_out_channels,
        model_kernel_size=cfg.model_kernel_size,
        model_hidden_dim=cfg.model_hidden_dim,
    )
    print("Torch log saved at:", log_file)
    return log_file


def run_pair(cfg: ExperimentConfig):
    """
    Run Custom & Torch with the same settings.
    Returns:
        custom_log, torch_log
    """
    custom_log = run_custom(cfg)
    torch_log = run_torch(cfg)
    return custom_log, torch_log


def run_optimizer_grid(base_cfg: ExperimentConfig):
    """
    Run a 2x2 grid:
        [Custom, Torch] x [SGD, Adam]
    Returns:
        dict with 4 log paths.
    """
    logs = {}

    # SGD config
    cfg_sgd = ExperimentConfig(
        optimizer_type="SGD",
        num_epochs=base_cfg.num_epochs,
        batch_size=base_cfg.batch_size,
        lr=base_cfg.lr,
        model_out_channels=base_cfg.model_out_channels,
        model_kernel_size=base_cfg.model_kernel_size,
        model_hidden_dim=base_cfg.model_hidden_dim,
        log_name="k3_c8_h128_SGD_lr0.001",
        log_path=base_cfg.log_path,
    )

    # Adam config
    cfg_adam = ExperimentConfig(
        optimizer_type="Adam",
        num_epochs=base_cfg.num_epochs,
        batch_size=base_cfg.batch_size,
        lr=base_cfg.lr,
        model_out_channels=base_cfg.model_out_channels,
        model_kernel_size=base_cfg.model_kernel_size,
        model_hidden_dim=base_cfg.model_hidden_dim,
        log_name="k3_c8_h128_Adam_lr0.001",
        log_path=base_cfg.log_path,
    )

    # 2x2 runs
    logs["custom_sgd"] = run_custom(cfg_sgd)
    logs["custom_adam"] = run_custom(cfg_adam)
    logs["torch_sgd"] = run_torch(cfg_sgd)
    logs["torch_adam"] = run_torch(cfg_adam)

    return logs

def visualize_pair(custom_log: str, torch_log: str):
    """
    Full comparison between Custom and PyTorch for a single optimizer.
    """
    print("\n📊 Custom vs PyTorch: Convergence & Generalization")
    visuals.plot_loss_comparison(custom_log, torch_log)
    visuals.plot_accuracy_comparison(custom_log, torch_log)

    print("\n📉 Training Stability: Gradient Norms")
    visuals.plot_gradient_norm(custom_log, window=50)
    visuals.plot_gradient_norm(torch_log, window=50)

    print("\n⏱️ Efficiency Frontier: Time vs Memory")
    visuals.plot_efficiency_frontier([custom_log, torch_log])

    print("\n🧪 System Benchmark (Param count, time, memory)")
    visuals.plot_system_benchmark(custom_log, torch_log)


def visualize_diagnostics(log_path: str, title_prefix: str = ""):
    """
    Single-model diagnostics: step-level loss, gradient norms, confusion matrix, etc.
    """
    print(f"\n🔍 Detailed Training Dynamics for {title_prefix}")
    visuals.plot_detailed_loss(
        log_path, window=100,
        title=f"{title_prefix}: Detailed Step Loss"
    )

    print(f"\n📉 Gradient Norm Stability for {title_prefix}")
    visuals.plot_gradient_norm(
        log_path, window=50,
        title=f"{title_prefix}: Gradient Norms"
    )

    print(f"\n📊 Confusion Matrix for {title_prefix}")
    visuals.plot_confusion_matrix(log_path)


def visualize_optimizer_grid(logs: dict):
    """
    Visualize 4 runs:
        custom_sgd, custom_adam, torch_sgd, torch_adam
    """
    print("\n📊 Generating 4-Set Comparative Visualization (Optimizers x Frameworks)...")

    all_logs = [
        logs["custom_sgd"],
        logs["custom_adam"],
        logs["torch_sgd"],
        logs["torch_adam"],
    ]

    # 1. Loss curves
    visuals.plot_multi_loss(
        all_logs,
        title="Training Loss Comparison Across Optimizers"
    )

    # 2. Accuracy curves
    visuals.plot_multi_accuracy(
        all_logs,
        title="Test Accuracy Comparison Across Optimizers"
    )

    # 3. Gradient norms
    visuals.plot_multi_grad_norm(
        all_logs,
        window=50,
        title="Gradient Norm Comparison Across Optimizers"
    )

    # 4. Time & memory
    visuals.plot_performance_bar(
        all_logs,
        title="Training Speed & Memory Comparison Across Optimizers"
    )