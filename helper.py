import tracemalloc
import json
import time

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