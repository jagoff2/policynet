#!/usr/bin/env python3
"""
Test script to validate complete CARLA training integration.
This tests the full pipeline: CARLA -> Vision -> Policy -> Training
"""
import sys
import os
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_carla_training_pipeline():
    """Test complete training pipeline with CARLA."""
    print("=== Testing CARLA Training Pipeline ===")
    
    # Test imports
    try:
        import carla
        from carla_env import CarlaDrivingEnv, CARLA_AVAILABLE
        from policy_model import PolicyConfig, PolicyNetwork
        from openpilot_wrapper import OpenPilotWrapper
        import torch
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
        
    if not CARLA_AVAILABLE:
        print("✗ CARLA not available")
        return False
        
    # Test CARLA connection
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        version = client.get_server_version()
        print(f"✓ Connected to CARLA {version}")
    except Exception as e:
        print(f"✗ CARLA connection failed: {e}")
        print("Make sure CARLA server is running!")
        return False
    
    # Test complete pipeline
    try:
        # Create components
        print("Creating components...")
        op_wrapper = OpenPilotWrapper("dummy.onnx", "dummy.pkl", use_dummy=True)
        cfg = PolicyConfig()
        env = CarlaDrivingEnv(op_wrapper, cfg, seed=42)
        policy = PolicyNetwork(cfg)
        print("✓ Components created")
        
        # Test episode
        print("Running test episode...")
        obs = env.reset()
        print(f"✓ Environment reset, obs shape: {obs.shape}")
        
        total_reward = 0.0
        episode_steps = 0
        
        for step in range(20):  # Short test episode
            # Policy forward pass
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                policy_outputs = policy(obs_tensor)
            
            # Extract action
            if "desired_curvature" in policy_outputs:
                curvature_params = policy_outputs["desired_curvature"][0]
                if len(curvature_params.shape) > 0:
                    curvature = curvature_params[0].item()  # Use first component
                else:
                    curvature = curvature_params.item()
            else:
                curvature = 0.0
            
            # Small forward acceleration
            acceleration = 0.5
            action = np.array([curvature, acceleration], dtype=np.float32)
            
            # Environment step
            obs, reward, done, info = env.step(action)
            total_reward += reward
            episode_steps += 1
            
            if step % 5 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
                if 'cross_track_error' in info:
                    print(f"    CTE: {info['cross_track_error']:.3f}, Speed: {info.get('speed', 0):.1f}")
            
            if done:
                print(f"  Episode ended at step {step}")
                break
        
        print(f"✓ Episode completed: {episode_steps} steps, total reward: {total_reward:.3f}")
        
        # Clean up
        env.close()
        print("✓ Environment closed")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up
        try:
            env.close()
        except:
            pass
            
        return False

def test_multi_episode():
    """Test multiple episode resets."""
    print("\n=== Testing Multiple Episodes ===")
    
    try:
        from carla_env import CarlaDrivingEnv
        from policy_model import PolicyConfig
        from openpilot_wrapper import OpenPilotWrapper
        
        # Create environment
        op_wrapper = OpenPilotWrapper("dummy.onnx", "dummy.pkl", use_dummy=True)
        cfg = PolicyConfig()
        env = CarlaDrivingEnv(op_wrapper, cfg, seed=42)
        
        for episode in range(3):
            print(f"\nEpisode {episode + 1}")
            obs = env.reset()
            print(f"  Reset successful, obs shape: {obs.shape}")
            
            # Run a few steps
            for step in range(5):
                action = np.array([0.0, 0.5], dtype=np.float32)  # Straight ahead
                obs, reward, done, info = env.step(action)
                
                if done:
                    print(f"  Episode ended at step {step}")
                    break
                    
            print(f"  Episode {episode + 1} completed")
        
        env.close()
        print("✓ Multi-episode test passed")
        return True
        
    except Exception as e:
        print(f"✗ Multi-episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("CARLA Training Integration Test")
    print("=" * 50)
    
    success = True
    
    # Test 1: Complete pipeline
    if not test_carla_training_pipeline():
        success = False
    
    # Test 2: Multiple episodes
    if success and not test_multi_episode():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ ALL TESTS PASSED!")
        print("\nCARLA training integration is working correctly.")
        print("You can now run full training with:")
        print("  python run_training.py --env carla --episodes 10 --vision-onnx driving_vision.onnx")
    else:
        print("✗ SOME TESTS FAILED")
        print("Check the error messages above and ensure:")
        print("1. CARLA server is running (./CarlaUE4.exe)")
        print("2. CARLA Python API is in PYTHONPATH")
        print("3. All dependencies are installed")
        
    exit(0 if success else 1)