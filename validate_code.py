#!/usr/bin/env python3
"""
Code structure validation without requiring external dependencies.
Checks syntax, imports, and basic structure.
"""

import ast
import os
import sys

def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def analyze_file_structure(filepath):
    """Analyze the structure of a Python file."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return True, {
            'classes': classes,
            'functions': functions, 
            'imports': imports
        }
    except Exception as e:
        return False, str(e)

def main():
    """Validate core Python files."""
    files_to_check = [
        'policy_model.py',
        'mdn.py',
        'openpilot_wrapper.py',
        'carla_env.py',
        'train.py'
    ]
    
    print("=== Code Structure Validation ===")
    
    all_valid = True
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"❌ {filepath}: File not found")
            all_valid = False
            continue
            
        # Check syntax
        syntax_ok, syntax_error = check_python_syntax(filepath)
        if not syntax_ok:
            print(f"❌ {filepath}: Syntax error - {syntax_error}")
            all_valid = False
            continue
        
        # Analyze structure
        struct_ok, struct_info = analyze_file_structure(filepath)
        if not struct_ok:
            print(f"❌ {filepath}: Structure analysis failed - {struct_info}")
            all_valid = False
            continue
        
        print(f"✅ {filepath}: Valid syntax")
        print(f"   Classes: {', '.join(struct_info['classes'][:5])}")
        print(f"   Functions: {', '.join(struct_info['functions'][:5])}")
        print(f"   Key imports: {len(struct_info['imports'])} total")
    
    # Check for openpilot compatibility markers
    print("\n=== Openpilot Compatibility Check ===")
    
    compatibility_markers = {
        'policy_model.py': [
            'PolicyConfig', 'PolicyNetwork', 'FEATURE_LEN', 'INPUT_HISTORY_BUFFER_LEN'
        ],
        'mdn.py': [
            'safe_exp', 'mdn_split_params', 'mdn_log_prob'
        ],
        'train.py': [
            'extract_action_and_log_prob', 'unpack_observation'
        ]
    }
    
    for filepath, markers in compatibility_markers.items():
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
        
        found_markers = []
        missing_markers = []
        
        for marker in markers:
            if marker in content:
                found_markers.append(marker)
            else:
                missing_markers.append(marker)
        
        print(f"📋 {filepath}: {len(found_markers)}/{len(markers)} compatibility markers")
        if missing_markers:
            print(f"   Missing: {', '.join(missing_markers)}")
    
    # Check metadata files
    print("\n=== Metadata Files Check ===")
    metadata_files = [
        'driving_vision_dtr_metadata.pkl',
        'driving_policy_dtr_metadata.pkl'
    ]
    
    for filepath in metadata_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {filepath}: {size} bytes")
        else:
            print(f"❌ {filepath}: Not found")
    
    print(f"\n=== Results ===")
    if all_valid:
        print("🎉 All Python files have valid syntax and structure!")
        print("📦 Ready for dependency installation and testing")
    else:
        print("❌ Some files have syntax or structure issues")
    
    print("\n=== Next Steps ===")
    print("1. Install dependencies: uv sync")  
    print("2. Test with dummy data: python train.py --env dummy --episodes 1")
    print("3. Validate ONNX export")
    print("4. Test with real vision model data")

if __name__ == "__main__":
    main()