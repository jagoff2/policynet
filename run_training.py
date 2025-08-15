#!/usr/bin/env python3
"""
Complete training pipeline runner for the openpilot-compatible policy model.
This script handles dependency checking, training, and deployment preparation.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    print("=== Checking Dependencies ===")
    
    required_modules = [
        ('torch', 'PyTorch for neural networks'),
        ('numpy', 'Numerical computing'),
        ('gymnasium', 'RL environment interface'),
    ]
    
    missing_modules = []
    
    for module_name, description in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}: Available")
        except ImportError:
            print(f"✗ {module_name}: Missing ({description})")
            missing_modules.append(module_name)
    
    if missing_modules:
        print(f"\nMissing dependencies: {', '.join(missing_modules)}")
        print("Install with: uv sync")
        return False
    
    return True

def validate_metadata_files():
    """Check that required metadata files exist."""
    print("\n=== Checking Metadata Files ===")
    
    required_files = [
        ('driving_vision_dtr_metadata.pkl', 'Vision model metadata'),
        ('driving_policy_dtr_metadata.pkl', 'Policy model reference metadata'),
    ]
    
    missing_files = []
    
    for filename, description in required_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename}: {size} bytes")
        else:
            print(f"✗ {filename}: Missing ({description})")
            missing_files.append(filename)
    
    if missing_files:
        print("\nNote: Missing metadata files will use dummy mode")
        print("For full vision integration, provide the openpilot model files")
    
    return len(missing_files) == 0

def run_training(args):
    """Execute the training pipeline."""
    print(f"\n=== Starting Training ===")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output_dir}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build training command
    cmd = [
        sys.executable, 'train.py',
        '--env', args.env,
        '--episodes', str(args.episodes),
        '--output-dir', args.output_dir,
        '--timesteps', str(args.timesteps),
        '--lr', str(args.lr),
        '--seed', str(args.seed),
    ]
    
    if args.vision_onnx and os.path.exists(args.vision_onnx):
        cmd.extend(['--vision-onnx', args.vision_onnx])
    else:
        print("Warning: No vision model file specified, using dummy features")
        
    if args.vision_meta and os.path.exists(args.vision_meta):
        cmd.extend(['--vision-meta', args.vision_meta])
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        return False

def validate_outputs(output_dir):
    """Validate that training produced expected outputs."""
    print(f"\n=== Validating Outputs ===")
    
    expected_files = [
        ('policy_model.pth', 'PyTorch model checkpoint'),
        ('driving_policy.onnx', 'ONNX model for openpilot'),
        ('driving_policy_metadata.pkl', 'Openpilot metadata'),
    ]
    
    all_present = True
    
    for filename, description in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filename}: {size} bytes ({description})")
        else:
            print(f"✗ {filename}: Missing ({description})")
            all_present = False
    
    return all_present

def show_deployment_instructions(output_dir):
    """Show instructions for deploying the trained model."""
    print(f"\n=== Deployment Instructions ===")
    print(f"Your trained model is ready in: {os.path.abspath(output_dir)}")
    print(f"")
    print(f"For openpilot deployment:")
    print(f"1. Copy files to openpilot model directory:")
    print(f"   cp {output_dir}/driving_policy.onnx /path/to/openpilot/models/")
    print(f"   cp {output_dir}/driving_policy_metadata.pkl /path/to/openpilot/models/")
    print(f"")
    print(f"2. Update openpilot to use the new model:")
    print(f"   - Modify selfdrive/modeld/modeld.py")
    print(f"   - Update MODEL_PATHS to include your model")
    print(f"")
    print(f"3. Test with openpilot:")
    print(f"   cd /path/to/openpilot")
    print(f"   ./tools/replay/replay /path/to/route/segment")
    print(f"")
    print(f"For further development:")
    print(f"   - Continue training: python run_training.py --episodes 100")
    print(f"   - Analyze performance: python analyze_logs.py {output_dir}")
    print(f"   - Fine-tune: python train.py --load {output_dir}/policy_model.pth")

def main():
    parser = argparse.ArgumentParser(description="Train openpilot-compatible policy model")
    
    # Training parameters
    parser.add_argument('--env', choices=['dummy', 'carla'], default='dummy',
                       help='Training environment')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of training episodes')
    parser.add_argument('--timesteps', type=int, default=500,
                       help='Max timesteps per episode')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Model files
    parser.add_argument('--vision-onnx', type=str,
                       help='Path to vision model ONNX file')
    parser.add_argument('--vision-meta', type=str,
                       help='Path to vision metadata pickle file')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Directory to save trained model')
    
    # Control flags
    parser.add_argument('--skip-deps-check', action='store_true',
                       help='Skip dependency checking')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    print("=== Openpilot Policy Training Pipeline ===")
    
    # Dependency check
    if not args.skip_deps_check and not check_dependencies():
        print("\\nPlease install dependencies before proceeding.")
        return 1
    
    # Metadata validation
    validate_metadata_files()
    
    if args.dry_run:
        print(f"\\nDry run - would train with:")
        print(f"  Environment: {args.env}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Output: {args.output_dir}")
        return 0
    
    # Run training
    if not run_training(args):
        return 1
    
    # Validate outputs
    if not validate_outputs(args.output_dir):
        print("\\nWarning: Some expected outputs are missing")
        return 1
    
    # Show deployment instructions
    show_deployment_instructions(args.output_dir)
    
    print(f"\\n🎉 Training pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())