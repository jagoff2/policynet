#!/usr/bin/env python3
"""
Test if the training pipeline is ready to run.
This checks all imports and basic functionality.
"""

import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_dependencies():
    """Test that required dependencies are available."""
    print("=== Testing Dependencies ===")
    
    all_good = True
    
    # Test core dependencies
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        all_good = False
        
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy not available: {e}")
        all_good = False
        
    try:
        import gymnasium
        print(f"✓ Gymnasium available")
    except ImportError as e:
        print(f"✗ Gymnasium not available: {e}")
        all_good = False
        
    return all_good

def test_openpilot_imports():
    """Test openpilot module imports."""
    print("\n=== Testing Openpilot Imports ===")
    
    # Add openpilot to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    openpilot_path = os.path.join(current_dir, 'openpilot')
    
    if not os.path.exists(openpilot_path):
        print(f"✗ Openpilot directory not found at {openpilot_path}")
        return False
        
    sys.path.insert(0, openpilot_path)
    print(f"Added {openpilot_path} to sys.path")
    
    try:
        from selfdrive.modeld.constants import ModelConstants
        print(f"✓ ModelConstants imported")
        print(f"  - FEATURE_LEN: {ModelConstants.FEATURE_LEN}")
        print(f"  - INPUT_HISTORY_BUFFER_LEN: {ModelConstants.INPUT_HISTORY_BUFFER_LEN}")
        print(f"  - PLAN_MHP_N: {ModelConstants.PLAN_MHP_N}")
    except ImportError as e:
        print(f"✗ Failed to import ModelConstants: {e}")
        return False
        
    try:
        from selfdrive.modeld.parse_model_outputs import Parser
        parser = Parser()
        print(f"✓ Parser imported and created")
    except ImportError as e:
        print(f"✗ Failed to import Parser: {e}")
        return False
        
    return True

def test_custom_modules():
    """Test our custom modules."""
    print("\n=== Testing Custom Modules ===")
    
    all_good = True
    
    try:
        from policy_model import PolicyConfig, PolicyNetwork
        cfg = PolicyConfig()
        print(f"✓ PolicyConfig: feature_len={cfg.feature_len}, plan_output_dim={cfg.plan_output_dim}")
        
        import torch
        model = PolicyNetwork(cfg)
        print(f"✓ PolicyNetwork created with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"✗ Policy model error: {e}")
        all_good = False
        
    try:
        from openpilot_wrapper import OpenPilotWrapper
        wrapper = OpenPilotWrapper("dummy.pkl", "dummy.pkl", use_dummy=True)
        print(f"✓ OpenPilotWrapper: feature_len={wrapper.feature_len}")
        
        features = wrapper.get_hidden_state({})
        print(f"✓ Generated features shape: {len(features) if isinstance(features, list) else features.shape}")
    except Exception as e:
        print(f"✗ OpenPilot wrapper error: {e}")
        all_good = False
        
    try:
        from carla_env import DummyDrivingEnv
        from policy_model import PolicyConfig
        from openpilot_wrapper import OpenPilotWrapper
        
        cfg = PolicyConfig()
        wrapper = OpenPilotWrapper("dummy.pkl", "dummy.pkl", use_dummy=True)
        env = DummyDrivingEnv(wrapper, cfg, seed=42)
        
        obs = env.reset()
        print(f"✓ DummyDrivingEnv: obs_shape={obs.shape}")
        
        import numpy as np
        action = np.array([0.1, 0.5])
        next_obs, reward, done, info = env.step(action)
        print(f"✓ Environment step: reward={reward:.3f}")
    except Exception as e:
        print(f"✗ Environment error: {e}")
        all_good = False
        
    return all_good

def test_training_components():
    """Test key training components."""
    print("\n=== Testing Training Components ===")
    
    all_good = True
    
    try:
        from train import extract_action_and_log_prob, unpack_observation
        print("✓ Training utilities imported")
    except Exception as e:
        print(f"✗ Training utilities error: {e}")
        all_good = False
        
    try:
        from mdn import safe_exp, mdn_split_params, mdn_mean
        print("✓ MDN utilities imported")
    except Exception as e:
        print(f"✗ MDN utilities error: {e}")
        all_good = False
        
    return all_good

def main():
    """Run all tests."""
    print("=== PolicyNet Training Readiness Check ===\n")
    
    # Run tests
    deps_ok = test_dependencies()
    openpilot_ok = test_openpilot_imports()
    custom_ok = test_custom_modules()
    training_ok = test_training_components()
    
    # Summary
    print("\n=== Summary ===")
    print(f"Dependencies: {'✓' if deps_ok else '✗'}")
    print(f"Openpilot imports: {'✓' if openpilot_ok else '✗'}")
    print(f"Custom modules: {'✓' if custom_ok else '✗'}")
    print(f"Training components: {'✓' if training_ok else '✗'}")
    
    if all([deps_ok, custom_ok, training_ok]):
        print("\n🎉 System is ready for training!")
        print("\nRun training with:")
        print("  python run_training.py --env dummy --episodes 50")
        
        if not openpilot_ok:
            print("\n⚠ Note: Openpilot imports failed but training can proceed with dummy mode")
        
        return 0
    else:
        print("\n❌ Some components need attention before training")
        return 1

if __name__ == "__main__":
    sys.exit(main())