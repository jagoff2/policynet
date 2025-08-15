#!/usr/bin/env python3
import sys
import os

def test_tinygrad_imports():
    print("=== Testing Tinygrad Import Setup ===")
    
    # Add tinygrad paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    openpilot_path = os.path.join(current_dir, 'openpilot')
    tinygrad_repo_path = os.path.join(openpilot_path, 'tinygrad_repo')
    tinygrad_symlink_path = os.path.join(openpilot_path, 'tinygrad')
    
    print(f"Current directory: {current_dir}")
    print(f"Openpilot path: {openpilot_path}")
    print(f"Tinygrad repo path: {tinygrad_repo_path}")
    print(f"Tinygrad symlink path: {tinygrad_symlink_path}")
    
    print(f"Tinygrad repo exists: {os.path.exists(tinygrad_repo_path)}")
    print(f"Tinygrad symlink exists: {os.path.exists(tinygrad_symlink_path)}")
    
    # Add to path
    if os.path.exists(tinygrad_repo_path) and tinygrad_repo_path not in sys.path:
        sys.path.insert(0, tinygrad_repo_path)
        print(f"✓ Added tinygrad_repo to path")
        
    if os.path.exists(tinygrad_symlink_path) and tinygrad_symlink_path not in sys.path:
        sys.path.insert(0, tinygrad_symlink_path)
        print(f"✓ Added tinygrad symlink to path")
    
    print(f"\nPython path includes:")
    for i, path in enumerate(sys.path[:10]):  # Show first 10 paths
        print(f"  {i}: {path}")
    
    # Test tinygrad imports
    print(f"\n=== Testing Tinygrad Imports ===")
    
    # Test 1: Basic tinygrad import
    try:
        import tinygrad
        print(f"✓ import tinygrad: SUCCESS")
        print(f"  tinygrad location: {tinygrad.__file__ if hasattr(tinygrad, '__file__') else 'No __file__'}")
    except Exception as e:
        print(f"✗ import tinygrad: FAILED - {e}")
    
    # Test 2: Tensor import
    try:
        from tinygrad.tensor import Tensor
        print(f"✓ from tinygrad.tensor import Tensor: SUCCESS")
    except Exception as e:
        print(f"✗ from tinygrad.tensor import Tensor: FAILED - {e}")
    
    # Test 3: The problematic uop import
    try:
        from tinygrad.uop import UOp
        print(f"✓ from tinygrad.uop import UOp: SUCCESS")
    except Exception as e:
        print(f"✗ from tinygrad.uop import UOp: FAILED - {e}")
        
    # Test 4: Check if tinygrad.uop module exists
    try:
        import tinygrad.uop
        print(f"✓ import tinygrad.uop: SUCCESS")
        print(f"  tinygrad.uop location: {tinygrad.uop.__file__ if hasattr(tinygrad.uop, '__file__') else 'No __file__'}")
    except Exception as e:
        print(f"✗ import tinygrad.uop: FAILED - {e}")
        
    # Test 5: List what's in tinygrad.uop
    try:
        import tinygrad.uop
        uop_items = [item for item in dir(tinygrad.uop) if not item.startswith('_')]
        print(f"  tinygrad.uop contains: {uop_items[:10]}...")  # Show first 10 items
    except Exception as e:
        print(f"  Could not list tinygrad.uop contents: {e}")

if __name__ == "__main__":
    test_tinygrad_imports()