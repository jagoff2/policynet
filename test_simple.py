#!/usr/bin/env python3
"""
Simple test script that checks what we can without requiring dependencies.
"""

import os
import sys
import importlib.util

def check_module_available(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def test_dependencies():
    """Test which dependencies are available."""
    print("=== Dependency Check ===")
    
    deps = {
        'torch': 'PyTorch (neural networks)',
        'numpy': 'NumPy (numerical computing)', 
        'gymnasium': 'Gymnasium (RL environments)',
        'onnx': 'ONNX (model export)',
    }
    
    available = {}
    
    for dep, desc in deps.items():
        if check_module_available(dep):
            print(f"✓ {dep}: Available")
            available[dep] = True
        else:
            print(f"✗ {dep}: Missing ({desc})")
            available[dep] = False
    
    return available

def test_file_structure():
    """Test that all expected files exist."""
    print("\n=== File Structure Check ===")
    
    required_files = [
        'policy_model.py',
        'mdn.py', 
        'openpilot_wrapper.py',
        'carla_env.py',
        'train.py',
        'run_training.py',
        'pyproject.toml'
    ]
    
    all_present = True
    
    for filename in required_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename}: {size} bytes")
        else:
            print(f"✗ {filename}: Missing")
            all_present = False
    
    return all_present

def test_imports_with_deps():
    """Test imports only if dependencies are available."""
    print("\n=== Import Testing (with dependencies) ===")
    
    if not check_module_available('torch'):
        print("⚠ Skipping import tests - PyTorch not available")
        return False
    
    try:
        from policy_model import PolicyConfig
        cfg = PolicyConfig()
        print(f"✓ PolicyConfig: feature_len={cfg.feature_len}")
        
        from mdn import safe_exp, mdn_split_params
        print("✓ MDN utilities imported successfully")
        
        from openpilot_wrapper import OpenPilotWrapper
        wrapper = OpenPilotWrapper("dummy.pkl", "dummy.pkl", use_dummy=True)
        print(f"✓ OpenPilotWrapper: feature_len={wrapper.feature_len}")
        
        if check_module_available('gymnasium'):
            from carla_env import DummyDrivingEnv
            env = DummyDrivingEnv(wrapper, cfg, seed=42)
            print(f"✓ DummyDrivingEnv: obs_shape={env.observation_space.low.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_without_deps():
    """Tests that work without dependencies."""
    print("\n=== No-Dependency Tests ===")
    
    # Test that files have reasonable content
    policy_content = ""
    if os.path.exists('policy_model.py'):
        with open('policy_model.py', 'r') as f:
            policy_content = f.read()
        
        markers = ['PolicyConfig', 'PolicyNetwork', 'export_to_onnx']
        found = [m for m in markers if m in policy_content]
        print(f"✓ policy_model.py: {len(found)}/{len(markers)} key markers found")
    
    # Test metadata files
    metadata_files = [
        'driving_vision_dtr_metadata.pkl',
        'driving_policy_dtr_metadata.pkl'
    ]
    
    metadata_ok = True
    for filepath in metadata_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filepath}: {size} bytes")
        else:
            print(f"✗ {filepath}: Missing")
            metadata_ok = False
    
    return metadata_ok

def main():
    """Run all tests."""
    print("=== Simple PolicyNet Test (No Mocks) ===")
    
    # Always run these
    deps_available = test_dependencies()
    files_ok = test_file_structure()
    basic_ok = test_without_deps()
    
    # Only run if dependencies available
    imports_ok = False
    if deps_available.get('torch', False):
        imports_ok = test_imports_with_deps()
    else:
        print("\n⚠ Skipping dependency-requiring tests")
        print("Install with: pip install torch numpy gymnasium onnx")
    
    print(f"\n=== Summary ===")
    print(f"File structure: {'✓' if files_ok else '✗'}")
    print(f"Basic content: {'✓' if basic_ok else '✗'}")
    print(f"Dependencies: {'✓' if all(deps_available.values()) else '✗'}")
    print(f"Imports: {'✓' if imports_ok else '⚠ (skipped)'}")
    
    if files_ok and basic_ok:
        print("\n🎉 Core structure is valid!")
        if all(deps_available.values()) and imports_ok:
            print("🚀 Ready for training!")
        else:
            print("📦 Install dependencies to start training")
    else:
        print("\n❌ Structure issues need fixing")

if __name__ == "__main__":
    main()