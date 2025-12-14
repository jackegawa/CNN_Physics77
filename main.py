"""
Entry Point for CNN_Physics77 Framework
=======================================

This script orchestrates the benchmarking experiments defined in the 
benchmarks/ directory. It supports single-run comparisons and grid searches.
"""

import argparse
from pathlib import Path

# Absolute imports based on project structure
from benchmarks.helper import (
    ExperimentConfig,
    run_pair,
    visualize_pair,
    visualize_diagnostics,
    run_optimizer_grid,
    visualize_optimizer_grid
)

def run_basic_benchmark(args):
    """Executes a 1v1 comparison between Custom CNN and PyTorch Baseline."""
    print("\n" + "="*60)
    print("RUNNING MODE: BASIC BENCHMARK (Custom vs PyTorch)")
    print("="*60)

    cfg = ExperimentConfig(
        optimizer_type=args.optimizer,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        log_name="basic_demo"
    )

    print(f"[Step 1/3] Training models (Opt={cfg.optimizer_type}, Epochs={cfg.num_epochs}, LR={cfg.lr}, Seed={cfg.seed})...")
    custom_log, torch_log = run_pair(cfg)

    print("\n[Step 2/3] Generating General Comparison Plots...")
    visualize_pair(custom_log, torch_log)

    print("\n[Step 3/3] Diagnostics...")
    if args.diagnostics:
        visualize_diagnostics(custom_log, title_prefix="Custom_CNN_Analysis")
        visualize_diagnostics(torch_log,  title_prefix="Torch_CNN_Analysis")
        print("Diagnostics generated for Custom + Torch.")
    else:
        print("Skipping diagnostics (use --diagnostics to enable).")

    print("\nBasic Benchmark Completed! Check './fig/' for generated plots.")

def run_grid_search(args):
    """Executes a 4-way Grid Search (SGD/Adam x Custom/Torch)."""
    print("\n" + "="*60)
    print("RUNNING MODE: OPTIMIZER GRID SEARCH (4-Way Comparison)")
    print("="*60)

    base_cfg = ExperimentConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        log_name="grid_demo"
    )

    print("[Step 1/2] Training 4 variants (SGD/Adam x Custom/Torch)...")
    logs = run_optimizer_grid(base_cfg)

    print("\n[Step 2/2] Generating Multi-Model Comparison Plots...")
    visualize_optimizer_grid(logs)

    print("\nGrid Search Completed! Check './fig/' for all generated plots.")

    if args.diagnostics:
        print("\n[Extra] Diagnostics for each run...")
        for name, log_path in logs.items():
            visualize_diagnostics(log_path, title_prefix=name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NumPy-CNN Benchmark Runner")

    parser.add_argument("--mode", type=str, default="basic", choices=["basic", "grid"], 
                        help="Select execution mode: 'basic' (1v1) or 'grid' (4-way search).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Input batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=67, help="Random seed for reproducibility.")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam"], 
                        help="Optimizer type (only used in basic mode).")
    parser.add_argument("--diagnostics", action="store_true", 
                        help="Enable detailed gradient/loss step logging.")

    args = parser.parse_args()

    # Ensure output directories exist
    Path("./fig").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)

    if args.mode == "basic":
        run_basic_benchmark(args)
        print("\nTIP: Try 'python main.py --mode grid --epochs 1' for a quick multi-optimizer check!")
    else:
        run_grid_search(args)