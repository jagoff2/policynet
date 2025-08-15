#!/usr/bin/env python3
"""
Install minimal dependencies for PolicyNet training.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def main():
    """Install core dependencies."""
    print("=== Installing PolicyNet Dependencies ===")
    
    # Core packages needed for training
    packages = [
        'torch',
        'numpy', 
        'gymnasium',
        'onnx',
        'opencv-python',
        'matplotlib'
    ]
    
    print("Installing packages:")
    for pkg in packages:
        print(f"  - {pkg}")
    
    print("\nProceeding with installation...")
    
    success_count = 0
    for package in packages:
        print(f"\nInstalling {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
            success_count += 1
        else:
            print(f"✗ {package} installation failed")
    
    print(f"\n=== Installation Summary ===")
    print(f"Successfully installed: {success_count}/{len(packages)} packages")
    
    if success_count == len(packages):
        print("🎉 All dependencies installed!")
        print("\nNow you can run:")
        print("  python test_simple.py    # Test the installation")
        print("  python run_training.py   # Start training")
    else:
        print("⚠ Some packages failed to install")
        print("You may need to install them manually")
    
    return success_count == len(packages)

if __name__ == "__main__":
    main()