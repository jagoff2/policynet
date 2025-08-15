#!/usr/bin/env python3
import sys
import os

def test_openpilot_imports():
    print("=== Testing Openpilot Import Setup ===")
    
    # Add openpilot to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    openpilot_path = os.path.join(current_dir, 'openpilot')
    
    print(f"Current directory: {current_dir}")
    print(f"Openpilot path: {openpilot_path}")
    print(f"Openpilot path exists: {os.path.exists(openpilot_path)}")
    
    if os.path.exists(openpilot_path) and openpilot_path not in sys.path:
        sys.path.insert(0, openpilot_path)
        print(f"✓ Added openpilot to sys.path")
    
    print(f"Python path includes:")
    for i, path in enumerate(sys.path[:10]):  # Show first 10 paths
        print(f"  {i}: {path}")
    
    # Test direct selfdrive imports
    print(f"\n=== Testing Direct Imports ===")
    
    # Test 1: Basic selfdrive import
    try:
        import selfdrive
        print(f"✓ import selfdrive: SUCCESS")
        print(f"  selfdrive location: {selfdrive.__file__ if hasattr(selfdrive, '__file__') else 'No __file__'}")
    except Exception as e:
        print(f"✗ import selfdrive: FAILED - {e}")
    
    # Test 2: Import modeld constants
    try:
        from selfdrive.modeld.constants import ModelConstants
        print(f"✓ from selfdrive.modeld.constants import ModelConstants: SUCCESS")
        print(f"  FEATURE_LEN: {ModelConstants.FEATURE_LEN}")
    except Exception as e:
        print(f"✗ from selfdrive.modeld.constants import ModelConstants: FAILED - {e}")
    
    # Test 3: Import parser
    try:
        from selfdrive.modeld.parse_model_outputs import Parser
        parser = Parser()
        print(f"✓ from selfdrive.modeld.parse_model_outputs import Parser: SUCCESS")
        print(f"  Parser instance created: {type(parser)}")
    except Exception as e:
        print(f"✗ from selfdrive.modeld.parse_model_outputs import Parser: FAILED - {e}")
    
    # Test 4: Check what files are actually in the openpilot/selfdrive directory
    print(f"\n=== Checking Openpilot Directory Structure ===")
    selfdrive_path = os.path.join(openpilot_path, 'selfdrive')
    print(f"Selfdrive path: {selfdrive_path}")
    print(f"Selfdrive path exists: {os.path.exists(selfdrive_path)}")
    
    if os.path.exists(selfdrive_path):
        modeld_path = os.path.join(selfdrive_path, 'modeld')
        print(f"Modeld path exists: {os.path.exists(modeld_path)}")
        
        if os.path.exists(modeld_path):
            files = os.listdir(modeld_path)
            print(f"Files in modeld directory:")
            for f in sorted(files):
                if f.endswith('.py') or f == '__pycache__':
                    print(f"  - {f}")
        else:
            print(f"Modeld directory not found!")
    else:
        print(f"Selfdrive directory not found!")
        # List what's actually in openpilot directory
        if os.path.exists(openpilot_path):
            contents = os.listdir(openpilot_path)
            print(f"Contents of openpilot directory:")
            for item in sorted(contents):
                if os.path.isdir(os.path.join(openpilot_path, item)):
                    print(f"  [DIR]  {item}")
                else:
                    print(f"  [FILE] {item}")

if __name__ == "__main__":
    test_openpilot_imports()