#!/usr/bin/env python3
"""
Debug script to test the CARLA->Vision pipeline step by step.
This helps isolate where the image processing is failing.
"""

import sys
import os
import numpy as np
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_carla_vision_pipeline():
    """Test the complete CARLA vision pipeline step by step."""
    print("=== Debug CARLA Vision Pipeline ===")
    
    try:
        # Test imports
        print("\n1. Testing imports...")
        import carla
        from carla_env import CarlaDrivingEnv, CARLA_AVAILABLE
        from openpilot_wrapper import OpenPilotWrapper
        from policy_model import PolicyConfig
        print("✓ All imports successful")
        
        if not CARLA_AVAILABLE:
            print("✗ CARLA not available")
            return False
        
        # Test CARLA connection
        print("\n2. Testing CARLA connection...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        version = client.get_server_version()
        print(f"✓ Connected to CARLA {version}")
        
        # Test OpenPilot wrapper creation
        print("\n3. Testing OpenPilot wrapper creation...")
        op_wrapper = OpenPilotWrapper(
            "./driving_vision.onnx",
            "./driving_vision_dtr_metadata.pkl",
            use_dummy=False  # Use real vision model
        )
        print(f"✓ OpenPilot wrapper created, feature_len={op_wrapper.feature_len}")
        
        # Test environment creation (will set up cameras)
        print("\n4. Testing environment creation...")
        cfg = PolicyConfig()
        env = CarlaDrivingEnv(op_wrapper, cfg, seed=42)
        print("✓ CARLA environment created")
        
        # Reset and get initial camera data
        print("\n5. Testing environment reset...")
        obs = env.reset()
        print(f"✓ Environment reset, obs shape: {obs.shape}")
        
        # Check camera data after reset
        print("\n6. Checking camera data...")
        env._wait_for_camera_data(timeout=2.0)
        
        if env.camera_data["main"] is not None:
            main_shape = env.camera_data["main"].shape
            print(f"✓ Main camera data: {main_shape}, dtype: {env.camera_data['main'].dtype}")
        else:
            print("✗ No main camera data")
            
        if env.camera_data["extra"] is not None:
            extra_shape = env.camera_data["extra"].shape
            print(f"✓ Extra camera data: {extra_shape}, dtype: {env.camera_data['extra'].dtype}")
        else:
            print("✗ No extra camera data")
        
        # Test vision feature extraction step by step
        print("\n7. Testing vision processing...")
        
        if env.camera_data["main"] is not None:
            main_img = env.camera_data["main"]
            extra_img = env.camera_data["extra"] if env.camera_data["extra"] is not None else main_img
            
            print(f"Raw images - Main: {main_img.shape} {main_img.dtype}, Extra: {extra_img.shape} {extra_img.dtype}")
            
            # Test preprocessing
            main_processed = (main_img.astype(np.float32) / 255.0)
            extra_processed = (extra_img.astype(np.float32) / 255.0)
            
            print(f"Processed images - Main: {main_processed.shape} {main_processed.dtype} [{main_processed.min():.3f}, {main_processed.max():.3f}]")
            print(f"                   Extra: {extra_processed.shape} {extra_processed.dtype} [{extra_processed.min():.3f}, {extra_processed.max():.3f}]")
            
            # Test vision wrapper input preparation
            imgs = {
                "road": main_processed,
                "big_road": extra_processed
            }
            
            print(f"Vision input dict keys: {list(imgs.keys())}")
            
            # Test vision model inputs
            print("\n8. Testing ONNX vision model...")
            try:
                # Get model input info
                model_inputs = op_wrapper.vision_model.get_inputs()
                print("ONNX model inputs:")
                for inp in model_inputs:
                    print(f"  - {inp.name}: shape {inp.shape}, type {inp.type}")
                
                # Test direct vision call
                features = op_wrapper.get_hidden_state(imgs)
                print(f"✓ Vision features extracted: shape {np.array(features).shape}, dtype: {type(features)}")
                print(f"  Feature stats: mean={np.mean(features):.4f}, std={np.std(features):.4f}")
                
            except Exception as e:
                print(f"✗ Vision extraction failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("✗ No camera data available for vision testing")
            return False
        
        # Clean up
        env.close()
        print("\n✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_carla_vision_pipeline()
    print(f"\nResult: {'SUCCESS' if success else 'FAILURE'}")
    sys.exit(0 if success else 1)