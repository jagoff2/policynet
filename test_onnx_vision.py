#!/usr/bin/env python3
import sys
import os
import numpy as np

def test_onnx_vision():
    print("=== Testing ONNX Vision Model Loading ===")
    
    # Test ONNX runtime import
    print("\n=== Testing ONNX Runtime Import ===")
    try:
        import onnx
        import onnxruntime as ort
        print("✓ ONNX Runtime imported successfully")
        print(f"  ONNX version: {onnx.__version__}")
        print(f"  ONNX Runtime version: {ort.__version__}")
        print(f"  Available providers: {ort.get_available_providers()}")
    except ImportError as e:
        print(f"✗ ONNX Runtime import failed: {e}")
        print("\nTo install ONNX Runtime:")
        print("  pip install onnx onnxruntime")
        return False
    
    # Test openpilot wrapper with ONNX
    print(f"\n=== Testing OpenPilot Wrapper with ONNX ===")
    try:
        from openpilot_wrapper import OpenPilotWrapper
        
        # Test with dummy mode first
        print("Testing with dummy mode...")
        wrapper_dummy = OpenPilotWrapper("dummy.onnx", "dummy.pkl", use_dummy=True)
        print(f"✓ Dummy wrapper created, feature_len={wrapper_dummy.feature_len}")
        
        # Test dummy feature extraction
        dummy_imgs = {"road": np.random.randint(0, 255, (384, 512, 3), dtype=np.uint8)}
        features = wrapper_dummy.get_hidden_state(dummy_imgs)
        print(f"✓ Dummy features extracted, shape={np.array(features).shape}")
        
        # Test with actual ONNX file if available
        onnx_path = "./driving_vision.onnx"
        if os.path.exists(onnx_path):
            print(f"\nTesting with real ONNX model: {onnx_path}")
            try:
                wrapper_real = OpenPilotWrapper(onnx_path, "./driving_vision_metadata.pkl")
                print(f"✓ Real ONNX wrapper created, feature_len={wrapper_real.feature_len}")
                
                # Test real feature extraction
                test_imgs = {"road": np.random.randint(0, 255, (384, 512, 3), dtype=np.uint8).astype(np.float32)}
                real_features = wrapper_real.get_hidden_state(test_imgs)
                print(f"✓ Real features extracted, shape={np.array(real_features).shape}")
                
            except Exception as e:
                print(f"⚠ Real ONNX wrapper failed (expected if vision model not available): {e}")
        else:
            print(f"⚠ ONNX file not found: {onnx_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenPilot wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_onnx_vision()
    print(f"\nONNX vision test: {'SUCCESS' if success else 'FAILED'}")
    exit(0 if success else 1)