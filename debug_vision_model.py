#!/usr/bin/env python3
import sys
import os
import pickle

def debug_vision_model_loading():
    print("=== Vision Model Loading Debug ===")
    
    # Add paths similar to openpilot_wrapper.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    openpilot_path = os.path.join(current_dir, 'openpilot')
    tinygrad_repo_path = os.path.join(openpilot_path, 'tinygrad_repo')
    
    if os.path.exists(tinygrad_repo_path) and tinygrad_repo_path not in sys.path:
        sys.path.insert(0, tinygrad_repo_path)
        print(f"✓ Added tinygrad_repo to path: {tinygrad_repo_path}")
        
    tinygrad_symlink_path = os.path.join(openpilot_path, 'tinygrad')
    if os.path.exists(tinygrad_symlink_path) and tinygrad_symlink_path not in sys.path:
        sys.path.insert(0, tinygrad_symlink_path)
        print(f"✓ Added tinygrad symlink to path: {tinygrad_symlink_path}")
        
    if os.path.exists(openpilot_path) and openpilot_path not in sys.path:
        sys.path.insert(0, openpilot_path)
        print(f"✓ Added openpilot path: {openpilot_path}")
    
    # Patch tinygrad first
    print(f"\\n=== Patching Tinygrad for Windows ===")
    try:
        qcom_file = os.path.join(tinygrad_repo_path, 'tinygrad', 'runtime', 'ops_qcom.py')
        if os.path.exists(qcom_file):
            with open(qcom_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if '# PATCHED_FOR_WINDOWS' not in content:
                original_assert = "assert sys.platform != 'win32'"
                patched_code = "# PATCHED_FOR_WINDOWS: assert sys.platform != 'win32'"
                
                if original_assert in content:
                    content = content.replace(original_assert, patched_code, 1)
                    with open(qcom_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✓ Patched QCOM runtime for Windows")
                else:
                    print(f"⚠ QCOM assertion not found")
            else:
                print(f"✓ QCOM runtime already patched")
        else:
            print(f"✗ QCOM runtime file not found")
    except Exception as e:
        print(f"⚠ Could not patch QCOM runtime: {e}")

    # Test tinygrad import
    print(f"\\n=== Testing Tinygrad Import ===")
    try:
        # Force CPU device to avoid Windows device issues
        os.environ['DEVICE'] = 'CPU'
        os.environ['CPU'] = '1'
        from tinygrad.tensor import Tensor
        print(f"✓ Tinygrad tensor import successful (using CPU device)")
    except Exception as e:
        print(f"✗ Tinygrad tensor import failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test vision model files
    vision_pkl_path = "./driving_vision_dtr_tinygrad.pkl"
    vision_meta_path = "./driving_policy_dtr_metadata.pkl"
    
    print(f"\\n=== Checking Vision Model Files ===")
    print(f"Vision PKL path: {vision_pkl_path}")
    print(f"Vision PKL exists: {os.path.exists(vision_pkl_path)}")
    if os.path.exists(vision_pkl_path):
        print(f"Vision PKL size: {os.path.getsize(vision_pkl_path)} bytes")
    
    print(f"Vision metadata path: {vision_meta_path}")
    print(f"Vision metadata exists: {os.path.exists(vision_meta_path)}")
    if os.path.exists(vision_meta_path):
        print(f"Vision metadata size: {os.path.getsize(vision_meta_path)} bytes")
    
    # Try to load vision metadata first
    print(f"\\n=== Loading Vision Metadata ===")
    try:
        with open(vision_meta_path, "rb") as f:
            metadata = pickle.load(f)
        print(f"✓ Vision metadata loaded successfully")
        print(f"  Metadata keys: {list(metadata.keys())}")
        if 'output_shapes' in metadata:
            print(f"  Output shapes: {metadata['output_shapes']}")
        if 'output_slices' in metadata:
            print(f"  Output slices keys: {list(metadata['output_slices'].keys())}")
    except Exception as e:
        print(f"✗ Vision metadata load failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    # Try to load vision model
    print(f"\\n=== Loading Vision Model ===")
    try:
        print(f"Attempting to load vision model...")
        with open(vision_pkl_path, "rb") as f:
            vision_model = pickle.load(f)
        print(f"✓ Vision model loaded successfully!")
        print(f"  Model type: {type(vision_model)}")
        print(f"  Model attributes: {[attr for attr in dir(vision_model) if not attr.startswith('_')][:10]}")
        
        # Try to inspect the model more
        if hasattr(vision_model, '__dict__'):
            print(f"  Model dict keys: {list(vision_model.__dict__.keys())[:5]}")
        
    except Exception as e:
        print(f"✗ Vision model load failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_vision_model_loading()