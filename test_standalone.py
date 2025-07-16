#!/usr/bin/env python3
"""
Simple test script to verify Fogo memory pair functionality.
This demonstrates the package can be used as a standalone component.
"""

import sys
import os

# Add the src directory to the Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from memory_pair import StreamNewtonMemoryPair
from l_bfgs import LimitedMemoryBFGS
from event_logging import init_logging

def test_memory_pair_basic():
    """Test basic memory pair functionality"""
    print("Testing StreamNewtonMemoryPair...")
    
    # Initialize the memory pair
    dim = 5
    memory_pair = StreamNewtonMemoryPair(
        dim=dim,
        lam=1.0,
        eps_total=1.0,
        delta_total=1e-5,
        max_deletions=5
    )
    
    # Test basic properties
    assert memory_pair.dim == dim
    assert memory_pair.lam == 1.0
    assert len(memory_pair.theta) == dim
    assert memory_pair.privacy_ok()
    
    print("✓ Memory pair initialization successful")
    
    # Test insertion of data points
    np.random.seed(42)
    X = np.random.randn(10, dim)
    y = np.random.randn(10)
    
    # Insert a few data points
    for i in range(3):
        memory_pair.insert(X[i], y[i])
    
    print("✓ Data insertion successful")
    
    # Test deletion (unlearning)
    if len(memory_pair.lbfgs.S) > 0:
        memory_pair.delete(X[0], y[0])
        print("✓ Data deletion (unlearning) successful")
    else:
        print("! Skipping deletion test - no curvature pairs available")
    
    # Verify privacy budget
    print(f"✓ Privacy budget OK: {memory_pair.privacy_ok()}")
    
    return True

def test_lbfgs_basic():
    """Test basic L-BFGS functionality"""
    print("\nTesting LimitedMemoryBFGS...")
    
    lbfgs = LimitedMemoryBFGS(m_max=5)
    
    # Test basic properties
    assert len(lbfgs) == 0
    
    # Add some curvature pairs
    np.random.seed(42)
    for i in range(3):
        s = np.random.randn(5)
        y = np.random.randn(5)
        lbfgs.add_pair(s, y)
    
    print(f"✓ Added {len(lbfgs)} curvature pairs")
    
    # Test direction computation
    g = np.random.randn(5)
    d = lbfgs.direction(g)
    
    assert d.shape == g.shape
    print("✓ Direction computation successful")
    
    return True

def test_standalone_usage():
    """Test that the package can be used as a standalone component"""
    print("\nTesting standalone usage...")
    
    # This simulates how someone would use the package after cloning
    # it into their repository
    
    # Create a simple machine unlearning scenario
    np.random.seed(42)
    dim = 10
    n_samples = 20
    
    # Generate synthetic data
    X = np.random.randn(n_samples, dim)
    y = X @ np.random.randn(dim) + 0.1 * np.random.randn(n_samples)
    
    # Initialize memory pair
    memory_pair = StreamNewtonMemoryPair(
        dim=dim,
        lam=0.1,
        eps_total=2.0,
        delta_total=1e-5,
        max_deletions=10
    )
    
    # Train on first half of data
    train_indices = list(range(n_samples // 2))
    for i in train_indices:
        memory_pair.insert(X[i], y[i])
    
    # Store initial parameters
    theta_initial = memory_pair.theta.copy()
    
    # "Unlearn" some data points
    unlearn_indices = train_indices[:3]
    for i in unlearn_indices:
        memory_pair.delete(X[i], y[i])
    
    # Verify parameters changed
    theta_after_unlearn = memory_pair.theta.copy()
    
    assert not np.allclose(theta_initial, theta_after_unlearn), "Parameters should change after unlearning"
    assert memory_pair.privacy_ok(), "Privacy budget should be maintained"
    
    print("✓ Standalone unlearning scenario successful")
    print(f"✓ Parameter change magnitude: {np.linalg.norm(theta_after_unlearn - theta_initial):.6f}")
    
    return True

if __name__ == "__main__":
    print("=== Fogo Memory Pair Standalone Test ===")
    print()
    
    try:
        # Initialize logging (optional, but shows it works)
        log_dir = init_logging()
        print(f"✓ Logging initialized (output dir: {log_dir})")
    except Exception as e:
        print(f"! Logging initialization failed: {e}")
        print("  (This is not critical for core functionality)")
    
    print()
    
    # Run tests
    success = True
    
    try:
        success &= test_memory_pair_basic()
        success &= test_lbfgs_basic() 
        success &= test_standalone_usage()
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print()
    if success:
        print("🎉 All tests passed! The package is ready for standalone use.")
        print()
        print("To use this package in another repository:")
        print("1. Clone or copy the src/ directory")
        print("2. Install dependencies: pip install numpy scipy structlog python-json-logger PyYAML")
        print("3. Import: from src.memory_pair import StreamNewtonMemoryPair")
        print("4. Use for machine unlearning experiments!")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        sys.exit(1)