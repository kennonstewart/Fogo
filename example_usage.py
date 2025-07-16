#!/usr/bin/env python3
"""
Example usage of Fogo memory pair for machine unlearning.

This demonstrates how to use the Fogo package in another repository
for machine unlearning experiments and comparison with other methods.
"""

import numpy as np
import sys
import os

# Add the src directory to Python path (when used as standalone)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the main classes
from memory_pair import StreamNewtonMemoryPair
from l_bfgs import LimitedMemoryBFGS

def generate_synthetic_data(n_samples=100, dim=10, noise_std=0.1):
    """Generate synthetic regression data"""
    np.random.seed(42)
    X = np.random.randn(n_samples, dim)
    true_theta = np.random.randn(dim)
    y = X @ true_theta + noise_std * np.random.randn(n_samples)
    return X, y, true_theta

def demonstrate_machine_unlearning():
    """Demonstrate machine unlearning with the memory pair"""
    print("=== Machine Unlearning Example ===")
    
    # Generate data
    X, y, true_theta = generate_synthetic_data()
    n_samples, dim = X.shape
    
    print(f"Generated {n_samples} samples with {dim} features")
    
    # Initialize memory pair
    memory_pair = StreamNewtonMemoryPair(
        dim=dim,
        lam=0.01,           # Ridge regularization
        eps_total=1.0,      # Privacy budget
        delta_total=1e-5,   # Privacy parameter
        max_deletions=20    # Maximum number of deletions
    )
    
    # Train on data
    print("\n1. Training on all data...")
    for i in range(n_samples):
        memory_pair.insert(X[i], y[i])
    
    theta_full = memory_pair.theta.copy()
    mse_full = np.mean((X @ theta_full - y)**2)
    print(f"   MSE on full data: {mse_full:.6f}")
    
    # Unlearn some data points
    forget_indices = [0, 1, 2, 5, 10, 15]  # Points to forget
    print(f"\n2. Unlearning {len(forget_indices)} data points...")
    
    for i in forget_indices:
        memory_pair.delete(X[i], y[i])
    
    theta_unlearned = memory_pair.theta.copy()
    
    # Evaluate unlearning
    remaining_indices = [i for i in range(n_samples) if i not in forget_indices]
    X_remaining = X[remaining_indices]
    y_remaining = y[remaining_indices]
    
    mse_unlearned = np.mean((X_remaining @ theta_unlearned - y_remaining)**2)
    print(f"   MSE on remaining data: {mse_unlearned:.6f}")
    
    # Compare with retraining from scratch
    print("\n3. Comparison with retraining from scratch...")
    
    memory_pair_retrain = StreamNewtonMemoryPair(
        dim=dim, lam=0.01, eps_total=1.0, delta_total=1e-5, max_deletions=20
    )
    
    for i in remaining_indices:
        memory_pair_retrain.insert(X[i], y[i])
    
    theta_retrain = memory_pair_retrain.theta.copy()
    mse_retrain = np.mean((X_remaining @ theta_retrain - y_remaining)**2)
    
    print(f"   MSE from retraining: {mse_retrain:.6f}")
    print(f"   Parameter difference: {np.linalg.norm(theta_unlearned - theta_retrain):.6f}")
    
    # Privacy status
    print(f"\n4. Privacy status:")
    print(f"   Privacy budget remaining: {memory_pair.eps_total - memory_pair.eps_spent:.6f}")
    print(f"   Privacy OK: {memory_pair.privacy_ok()}")
    
    return {
        'theta_full': theta_full,
        'theta_unlearned': theta_unlearned,
        'theta_retrain': theta_retrain,
        'mse_full': mse_full,
        'mse_unlearned': mse_unlearned,
        'mse_retrain': mse_retrain
    }

def demonstrate_comparison_framework():
    """Show how this can be used for comparing unlearning methods"""
    print("\n=== Comparison Framework Example ===")
    
    # This is how researchers could use this package to compare
    # different unlearning methods
    
    X, y, true_theta = generate_synthetic_data(n_samples=50, dim=5)
    
    print("Comparing different unlearning approaches:")
    
    # Method 1: Fogo Memory Pair
    print("\n1. Fogo Memory Pair:")
    memory_pair = StreamNewtonMemoryPair(dim=5, lam=0.1, max_deletions=10)
    
    # Train
    for i in range(len(X)):
        memory_pair.insert(X[i], y[i])
    
    # Unlearn first 5 points
    forget_set = list(range(5))
    for i in forget_set:
        memory_pair.delete(X[i], y[i])
    
    theta_fogo = memory_pair.theta.copy()
    print(f"   Final parameters norm: {np.linalg.norm(theta_fogo):.6f}")
    
    # Method 2: Retraining (baseline)
    print("\n2. Retraining (baseline):")
    memory_pair_baseline = StreamNewtonMemoryPair(dim=5, lam=0.1, max_deletions=10)
    
    remaining_indices = [i for i in range(len(X)) if i not in forget_set]
    for i in remaining_indices:
        memory_pair_baseline.insert(X[i], y[i])
    
    theta_baseline = memory_pair_baseline.theta.copy()
    print(f"   Final parameters norm: {np.linalg.norm(theta_baseline):.6f}")
    
    # Method 3: Naive approach (zero out - for comparison)
    print("\n3. Naive approach (for comparison):")
    theta_naive = np.zeros(5)
    print(f"   Final parameters norm: {np.linalg.norm(theta_naive):.6f}")
    
    # Comparison metrics
    print("\n4. Comparison metrics:")
    print(f"   Distance Fogo vs Baseline: {np.linalg.norm(theta_fogo - theta_baseline):.6f}")
    print(f"   Distance Fogo vs Naive: {np.linalg.norm(theta_fogo - theta_naive):.6f}")
    
    return {
        'fogo': theta_fogo,
        'baseline': theta_baseline,
        'naive': theta_naive
    }

if __name__ == "__main__":
    print("Fogo Memory Pair - Machine Unlearning Example")
    print("=" * 50)
    
    # Demonstrate basic unlearning
    results = demonstrate_machine_unlearning()
    
    # Demonstrate comparison framework
    comparison = demonstrate_comparison_framework()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("This package provides a ready-to-use memory pair implementation")
    print("for machine unlearning research. It can be easily integrated")
    print("into other repositories for comparison with different methods.")
    print()
    print("Key features:")
    print("- StreamNewtonMemoryPair: Main unlearning algorithm")
    print("- LimitedMemoryBFGS: Optimization backend")
    print("- Privacy-preserving deletion with (ε,δ) guarantees")
    print("- Event-based logging for experimental tracking")
    print("- Standalone package - just copy src/ directory!")