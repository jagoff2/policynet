#!/usr/bin/env python3
"""
Test openpilot path resolution and module loading.
"""

import os
import sys

def test_openpilot_path():
    """Test finding and loading openpilot modules."""
    print("=== Openpilot Path Test ===")
    
    # Check current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current script directory: {current_dir}")
    
    # Check for openpilot subdirectory
    openpilot_path = os.path.join(current_dir, 'openpilot')
    print(f"Looking for openpilot at: {openpilot_path}")
    
    if os.path.exists(openpilot_path):
        print("✓ Openpilot directory exists")
        
        # Check for key files
        selfdrive_path = os.path.join(openpilot_path, 'selfdrive')
        constants_file = os.path.join(selfdrive_path, 'modeld', 'constants.py')
        parser_file = os.path.join(selfdrive_path, 'modeld', 'parse_model_outputs.py')
        
        print(f"  selfdrive path: {selfdrive_path} - {'exists' if os.path.exists(selfdrive_path) else 'missing'}")
        print(f"  constants.py: {'exists' if os.path.exists(constants_file) else 'missing'}")
        print(f"  parse_model_outputs.py: {'exists' if os.path.exists(parser_file) else 'missing'}")
        
        # Try adding to path and importing
        if openpilot_path not in sys.path:
            sys.path.insert(0, openpilot_path)
            print(f"✓ Added {openpilot_path} to sys.path")
        
        # Test import
        try:
            from openpilot.selfdrive.modeld.constants import ModelConstants
            print("✓ Successfully imported ModelConstants")
            print(f"  FEATURE_LEN: {ModelConstants.FEATURE_LEN}")
            print(f"  INPUT_HISTORY_BUFFER_LEN: {ModelConstants.INPUT_HISTORY_BUFFER_LEN}")
            print(f"  PLAN_MHP_N: {ModelConstants.PLAN_MHP_N}")
        except ImportError as e:
            print(f"✗ Failed to import ModelConstants: {e}")
        
        try:
            from openpilot.selfdrive.modeld.parse_model_outputs import Parser
            parser = Parser()
            print("✓ Successfully imported and created Parser")
        except ImportError as e:
            print(f"✗ Failed to import Parser: {e}")
            
    else:
        print("✗ Openpilot directory not found")
        
        # List what's actually in the current directory
        print("Contents of current directory:")
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")

def test_wrapper_import():
    """Test importing the openpilot wrapper."""
    print("\n=== Wrapper Import Test ===")
    
    try:
        from openpilot_wrapper import OpenPilotWrapper, OPENPILOT_AVAILABLE
        print("✓ Successfully imported OpenPilotWrapper")
        print(f"  OPENPILOT_AVAILABLE: {OPENPILOT_AVAILABLE}")
        
        # Test creating wrapper in dummy mode
        wrapper = OpenPilotWrapper("dummy.pkl", "dummy.pkl", use_dummy=True)
        print(f"  Created wrapper with feature_len: {wrapper.feature_len}")
        
        # Test feature extraction
        features = wrapper.get_hidden_state({})
        print(f"  Generated dummy features shape: {features.shape}")
        
    except Exception as e:
        print(f"✗ Failed to import or test wrapper: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_openpilot_path()
    test_wrapper_import()