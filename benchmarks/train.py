import numpy as np
import time
import random
import tracemalloc
import os
from contextlib import redirect_stdout

from core.data import load_mnist
from core.model import CNN
from core.operations import SoftMaxOp
from core.tensor import Tensor
from core.optim import Adam, SGD

# Note: Relative import here assumes this is run as a module or via main.py
from .helper import get_memory_snapshot, log_to_json

def train(
    optimizer_type: str, 
    log_name: str, 
    log_path: str = "./logs",
    num_epochs: int = 5, 
    batch_size: int = 64, 
    lr: float = 0.001, 
    seed: int = 67, 
    model_out_channels: int = 8,
    model_kernel_size: int = 3,
    model_hidden_dim: int = 128,
    beta1: float = 0.9, 
    beta2: float = 0.999, 
    epsilon: float = 1e-8
):
    """
    Main training loop for the Custom CNN.
    """
    # 1. Reproducibility
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"{log_name}_k{model_kernel_size}_c{model_out_channels}_h{model_hidden_dim}_{optimizer_type}_lr{lr}_{timestamp_str}"
    log_filename = f"{log_path}/{base_filename}.json"
    txt_filename = f"{log_path}/{base_filename}.txt"

    print(f"[INFO] Training Start | Optimizer: {optimizer_type} | Epochs: {num_epochs}")
    
    with open(txt_filename, 'w') as f_log:
        with redirect_stdout(f_log):
            import warnings
            # Suppress specific numpy warnings for cleaner logs
            warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
            warnings.filterwarnings("ignore", message="overflow encountered in matmul")
            warnings.filterwarnings("ignore", message="invalid value encountered in matmul")
            
            tracemalloc.start()

            # 2. Load Data
            print(f"[INFO] Loading MNIST dataset...", end=" ")
            x_train, y_train, x_test, y_test = load_mnist()
            n_samples = x_train.shape[0]
            n_test = x_test.shape[0]
            print(f"Done. (Train: {n_samples}, Test: {n_test})")

            cur_mem, peak_mem = get_memory_snapshot()
            print(f"[INFO] Initial Memory: {cur_mem:.2f} MB | Peak Memory: {peak_mem:.2f} MB")
            
            # 3. Build Model & Optimizer
            model = CNN(
                out_channels=model_out_channels,
                kernel_size=model_kernel_size,
                hidden_dim=model_hidden_dim
            )
            softmax = SoftMaxOp()
            param_count = sum(p.data.size for p in model.params())
            print(f"[INFO] Model initialized with {param_count} parameters.")
            
            if optimizer_type == "Adam":
                optim = Adam(model.params(), lr=lr, beta1=beta1, beta2=beta2, eps=epsilon)
            elif optimizer_type == "SGD":
                optim = SGD(model.params(), lr=lr)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

            log_data = {
                "per_epoch_stats": [],
                "detailed_loss": [],
                "detailed_grad_norm": [],
                "final_evaluation": {}
            }

            # 4. Training Loop
            total_start = time.time()
            
            for epoch in range(num_epochs):
                epoch_start = time.time()
                print(f"\n[TRAIN] EPOCH {epoch+1}/{num_epochs}")
                print(f"-"*60)

                # Shuffle
                idx = np.random.permutation(n_samples)
                x_train, y_train = x_train[idx], y_train[idx]

                running_loss = 0.0
                n_batches = int(np.ceil(n_samples / batch_size))
                epoch_grad_norms = []
                
                # Batch Iteration
                for step, i in enumerate(range(0, n_samples, batch_size)):
                    end = min(i + batch_size, n_samples)
                    x_batch = x_train[i:end]
                    y_batch = y_train[i:end]
                    
                    if x_batch.shape[0] == 0: continue

                    # A. Forward Pass
                    x = Tensor(x_batch)
                    logits = model.forward(x)
                    loss = softmax(logits, y_batch)
                    
                    # B. Backward Pass
                    optim.zero_grad()
                    loss.backward()
                    
                    # C. Monitor Gradients
                    total_norm = 0.0
                    for p in model.params():
                        if p.grad is not None:
                            total_norm += np.sum(p.grad ** 2)
                    total_norm = np.sqrt(total_norm)
                    epoch_grad_norms.append(total_norm)
                    log_data["detailed_grad_norm"].append(float(total_norm))

                    # D. Update
                    optim.step()
                    
                    # Logging
                    loss_val = float(loss.data)
                    running_loss += loss_val * x_batch.shape[0]
                    log_data["detailed_loss"].append(loss_val)

                    # Console Log (Periodic)
                    if step % 50 == 0 or step == n_batches - 1:
                        cur_m, peak_m = get_memory_snapshot()
                        progress = (step + 1) / n_batches * 100
                        print(f"   Step {step+1:3d}/{n_batches} ({progress:5.1f}%) | "
                            f"Loss: {loss_val:.4f} | Mem: {cur_m:.1f} MB")

                # Epoch Summary
                avg_loss = running_loss / n_samples
                avg_grad = np.mean(epoch_grad_norms)
                epoch_time = time.time() - epoch_start
                
                # --- Evaluation ---
                print(f"[EVAL]  Validating...", end="\r")
                correct = 0
                collect_preds = (epoch == num_epochs - 1)
                all_preds = [] if collect_preds else None
                all_labels = [] if collect_preds else None

                test_batch_size = 1000  # Large batch for inference
                for j in range(0, n_test, test_batch_size):
                    end = min(j + test_batch_size, n_test)
                    x_b = Tensor(x_test[j:end])
                    out = model.forward(x_b)
                    pred = np.argmax(out.data, axis=1)
                    correct += np.sum(pred == y_test[j:end])

                    if collect_preds:
                        all_preds.extend(pred.tolist())
                        all_labels.extend(y_test[j:end].tolist())
                
                acc = correct / n_test
                _, final_peak = get_memory_snapshot()

                print(f"[RESULT] Epoch {epoch+1} Finished in {epoch_time:.2f}s | "
                    f"Train Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}% | Peak Mem: {final_peak:.1f} MB")

                log_data["per_epoch_stats"].append({
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "accuracy": acc,
                    "time": epoch_time,
                    "memory_peak": final_peak,
                    "avg_grad_norm": avg_grad
                })

                if collect_preds:
                    log_data["final_evaluation"] = {
                        "y_true": all_labels,
                        "y_pred": all_preds
                    }

            # 5. Finalize
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
                optimizer=optim
            )
            
            tracemalloc.stop()
    print(f"[INFO] Training log saved to: {log_file}")
    return log_file