#!/usr/bin/env python3
import sys
import os
import numpy as np
import time

def test_carla_connection():
    """Test connection to CARLA server."""
    print("=== Testing CARLA Connection ===")
    
    try:
        import carla
        print("✓ CARLA Python API imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CARLA: {e}")
        print("\nTo fix this:")
        print("1. Make sure CARLA 0.10.0 is installed")
        print("2. Add CARLA Python API to your PYTHONPATH:")
        print("   export PYTHONPATH=$PYTHONPATH:/path/to/CARLA_0.10.0/PythonAPI/carla/dist/carla-0.10.0-py3.x-xxx-xxx.egg")
        return False
    
    # Test connection to CARLA server
    try:
        print("Attempting to connect to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Test connection
        version = client.get_server_version()
        print(f"✓ Connected to CARLA server version: {version}")
        
        # Get world info
        world = client.get_world()
        world_map = world.get_map()
        spawn_points = world_map.get_spawn_points()
        
        print(f"✓ Current map: {world_map.name}")
        print(f"✓ Available spawn points: {len(spawn_points)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to connect to CARLA server: {e}")
        print("\nTo fix this:")
        print("1. Start CARLA server with: ./CarlaUE4.exe (or CarlaUE4.sh on Linux)")
        print("2. Make sure server is running on localhost:2000")
        print("3. Check firewall settings")
        return False

def test_carla_environment():
    """Test the CARLA driving environment."""
    print("\n=== Testing CARLA Environment ===")
    
    try:
        # Import required modules
        from carla_env import CarlaDrivingEnv, CARLA_AVAILABLE
        from policy_model import PolicyConfig
        from openpilot_wrapper import OpenPilotWrapper
        
        if not CARLA_AVAILABLE:
            print("✗ CARLA not available")
            return False
            
        print("✓ Imported CARLA environment successfully")
        
        # Create dummy openpilot wrapper
        op_wrapper = OpenPilotWrapper("dummy.onnx", "dummy.pkl", use_dummy=True)
        print("✓ Created OpenPilot wrapper (dummy mode)")
        
        # Create policy config
        cfg = PolicyConfig()
        print("✓ Created policy config")
        
        # Create CARLA environment
        print("Creating CARLA environment...")
        env = CarlaDrivingEnv(op_wrapper, cfg, seed=42)
        print("✓ CARLA environment created successfully")
        
        # Test reset
        print("Resetting environment...")
        obs = env.reset()
        print(f"✓ Environment reset, observation shape: {obs.shape}")
        
        # Test a few steps
        print("Testing environment steps...")
        for i in range(5):
            # Random action: [curvature, acceleration]
            action = np.array([
                np.random.uniform(-0.5, 0.5),  # Curvature
                np.random.uniform(-1.0, 1.0)   # Acceleration
            ])
            
            obs, reward, done, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.3f}, done={done}, info keys: {list(info.keys())}")
            
            if done:
                print("    Episode terminated")
                break
                
            # Small delay to see the vehicle moving
            time.sleep(0.1)
        
        # Clean up
        env.close()
        print("✓ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ CARLA environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_with_carla():
    """Test training pipeline with CARLA."""
    print("\n=== Testing Training with CARLA ===")
    
    try:
        # Import training modules
        from carla_env import CarlaDrivingEnv
        from policy_model import PolicyConfig, PolicyNetwork
        from openpilot_wrapper import OpenPilotWrapper
        import torch
        
        print("✓ Imported training modules")
        
        # Create components
        op_wrapper = OpenPilotWrapper("dummy.onnx", "dummy.pkl", use_dummy=True)
        cfg = PolicyConfig()
        env = CarlaDrivingEnv(op_wrapper, cfg, seed=42)
        
        # Create policy network
        policy = PolicyNetwork(cfg)
        print("✓ Created policy network")
        
        # Test single training step
        obs = env.reset()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dim
        
        with torch.no_grad():
            policy_outputs = policy(obs_tensor)
        
        print("✓ Policy forward pass successful")
        print(f"  Policy outputs keys: {list(policy_outputs.keys())}")
        
        # Extract action (simplified)
        if "desired_curvature" in policy_outputs:
            curvature = policy_outputs["desired_curvature"][0, 0].item()  # MDN mean
        else:
            curvature = 0.0
            
        action = np.array([curvature, 0.5])  # Small acceleration
        
        obs, reward, done, info = env.step(action)
        print(f"✓ Training step completed: reward={reward:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("CARLA Environment Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test 1: CARLA connection
    if not test_carla_connection():
        success = False
        print("\nSkipping further tests due to CARLA connection failure")
    else:
        # Test 2: CARLA environment
        if not test_carla_environment():
            success = False
        
        # Test 3: Training integration
        if not test_training_with_carla():
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ ALL TESTS PASSED - CARLA integration is working!")
        print("\nYou can now run training with:")
        print("  python run_training.py --env carla --episodes 10")
    else:
        print("✗ SOME TESTS FAILED - Check the errors above")
        
    exit(0 if success else 1)