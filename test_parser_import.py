#!/usr/bin/env python3
import sys
import os

def test_parser_import():
    print("=== Testing Parser Import After Fix ===")
    
    # Add openpilot to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    openpilot_path = os.path.join(current_dir, 'openpilot')
    
    if os.path.exists(openpilot_path) and openpilot_path not in sys.path:
        sys.path.insert(0, openpilot_path)
        print(f"✓ Added openpilot to sys.path: {openpilot_path}")
    
    # Test the Parser import that was failing
    try:
        print("Attempting to import Parser...")
        from selfdrive.modeld.parse_model_outputs import Parser
        print("✓ Parser import successful!")
        
        # Try to create a Parser instance
        parser = Parser()
        print("✓ Parser instance created successfully!")
        print(f"  Parser type: {type(parser)}")
        
        # Check if Parser has expected methods
        methods = [attr for attr in dir(parser) if not attr.startswith('_')]
        print(f"  Parser methods: {methods[:5]}...")  # Show first 5 methods
        
        return True
        
    except Exception as e:
        print(f"✗ Parser import failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        
        # Try to provide more debug info
        try:
            import selfdrive.modeld.parse_model_outputs
            print(f"  Module loaded: {selfdrive.modeld.parse_model_outputs}")
        except Exception as e2:
            print(f"  Could not load module: {e2}")
        
        return False

if __name__ == "__main__":
    success = test_parser_import()
    print(f"\nParser import test: {'SUCCESS' if success else 'FAILED'}")
    exit(0 if success else 1)