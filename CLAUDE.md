# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a policy network training codebase for openpilot-compatible autonomous driving models using reinforcement learning. The system trains neural networks that can produce trajectory planning and control outputs compatible with comma.ai's openpilot stack.

**Key Components:**
- `train.py` - Main training entry point with PPO implementation
- `policy_model.py` - PyTorch-based recurrent neural network architecture 
- `openpilot_wrapper.py` - Interface to openpilot's vision model and parsing utilities
- `carla_env.py` - Driving simulation environments (CARLA-based and dummy)
- `mdn.py` - Mixture density network utilities for probabilistic outputs

## Development Commands

### Quick Start Training
```bash
# Recommended: Use the training runner (handles dependencies and validation)
python run_training.py --env dummy --episodes 50

# Direct training (requires manual dependency management)
python train.py --env dummy --episodes 10 --output-dir ./output

# Full training pipeline with all checks
python run_training.py \
    --env dummy \
    --episodes 100 \
    --timesteps 1000 \
    --lr 3e-4 \
    --output-dir ./trained_models
```

### Advanced Training
```bash
# With real vision model (if available)
python run_training.py \
    --env dummy \
    --vision-onnx driving_vision.onnx \
    --vision-meta driving_vision_metadata.pkl \
    --episodes 200

# CARLA training (requires CARLA 0.10.0 installation and server running)
python run_training.py --env carla --episodes 500

# Development mode with validation
python run_training.py --dry-run  # Preview without training
python validate_code.py           # Check code structure
```

### Testing and Validation
```bash
# Code structure validation (no dependencies required)
python validate_code.py

# Pipeline testing with mocked dependencies  
python test_pipeline.py

# Full training test (requires dependencies)
python run_training.py --episodes 5 --dry-run

# From openpilot/ subdirectory (if using openpilot tools):
cd openpilot && pytest
cd openpilot && ruff check .
```

### Dependencies
```bash
# Install core dependencies
uv sync  # Recommended package manager
# OR
pip install torch numpy gymnasium onnx opencv-python matplotlib

# Optional dependencies
# CARLA: Requires manual installation (see CARLA Setup below)
# ONNX Runtime: For real vision model integration 
# Openpilot: Full codebase available in openpilot/ subdirectory
```

**Dependency Layers:**
- **Core**: PyTorch, numpy, gymnasium (required for training)
- **Vision**: ONNX Runtime, openpilot modules (optional, falls back to dummy features)
- **Simulation**: CARLA (optional, dummy environment available)
- **Export**: ONNX (required for openpilot deployment)

### CARLA Setup (Optional)

For realistic simulation training, you can use CARLA 0.10.0:

```bash
# 1. Download CARLA 0.10.0
wget https://github.com/carla-simulator/carla/releases/download/0.10.0/CARLA_0.10.0.tar.gz
tar -xzf CARLA_0.10.0.tar.gz

# 2. Start CARLA server (in terminal 1)
cd CARLA_0.10.0
./CarlaUE4.exe -quality-level=Low -resx=800 -resy=600

# 3. Set up Python API (in terminal 2)
export PYTHONPATH=$PYTHONPATH:$(pwd)/PythonAPI/carla/dist/carla-0.10.0-py3.x-win-amd64.egg

# 4. Test CARLA connection
python test_carla_env.py

# 5. Run training with CARLA
python run_training.py --env carla --episodes 50
```

**CARLA Requirements:**
- CARLA 0.10.0 (compatible with Python 3.7-3.10)
- Windows 10/11 or Ubuntu 18.04+
- DirectX 11 or Vulkan support
- 8GB+ RAM, dedicated GPU recommended

**Notes:**
- CARLA server must be running before starting training
- Use `--quality-level=Low` for faster training
- Training automatically randomizes weather and maps
- Supports collision detection and lane-keeping metrics

## Architecture

### Policy Network Design
- **Recurrent Architecture**: GRU-based network maintaining hidden state across timesteps
- **Multi-head Output**: 
  - Plan trajectory (33 timesteps × 15 dimensions) using mixture density networks (MDN)
  - Desired curvature (single value) using MDN
  - Desire state classification (8 classes for turn signals, lane changes)
- **Input Processing**: Vision features + desire history + auxiliary parameters (traffic convention, lateral control params)

### Training Pipeline
1. **Environment**: Either CARLA simulator or lightweight dummy kinematic simulation
2. **Vision Processing**: Uses openpilot's vision model to extract features from camera images
3. **RL Algorithm**: Proximal Policy Optimization (PPO) with Generalized Advantage Estimation
4. **Action Space**: Continuous (desired_curvature, desired_acceleration)
5. **Export**: Trained models export to ONNX for openpilot deployment

### File Structure
```
├── train.py              # Main training script
├── policy_model.py       # Neural network architecture  
├── openpilot_wrapper.py  # Vision model interface
├── carla_env.py          # Simulation environments
├── mdn.py               # Mixture density utilities
└── openpilot/           # Full openpilot codebase (submodule)
    ├── pyproject.toml   # Python dependencies
    ├── selfdrive/       # Core autonomous driving logic
    ├── system/          # Hardware/system management
    └── tools/           # Development utilities
```

### Key Concepts

### Openpilot Interface Contract
- **Inputs**: `features_buffer` (vision), `desire` (maneuver intent), `traffic_convention` (LHD/RHD), `lateral_control_params` (velocity, delay), `prev_desired_curv` (control history)
- **Outputs**: `plan` (trajectory MDN), `desired_curvature` (steering MDN), `desire_state` (maneuver classification)
- **Frequency**: 20Hz operation matching openpilot's control loop
- **Format**: ONNX model + metadata pickle for tinygrad runtime

### Training Philosophy
- **No Curriculum**: Full randomization from start (following "random pipes" success)
- **Hard Constraints**: Immediate episode termination on violations
- **Risk Sensitivity**: CVaR-style reward to eliminate tail risks
- **Domain Randomization**: Physics, sensors, scenarios vary each episode

### Architecture Principles
- **Lean Recurrent**: GRU backbone with layernorm and proper initialization
- **MDN Native**: Direct mixture density outputs matching parser expectations
- **Latency Aware**: Uses live `lat_delay` parameter for control timing
- **Deployment Ready**: ONNX export with shape validation and compatibility checks

## Development Notes

### Training
- **GPU/CPU**: Supports both, defaults to CPU for compatibility
- **Environments**: Dummy kinematic simulation (fast) or CARLA (realistic)
- **Vision Integration**: Real openpilot vision model or dummy random features
- **Early Stopping**: Automatically stops when reasonable performance is reached

### Model Export
- **Format**: ONNX + metadata pickle for openpilot compatibility
- **Validation**: Automatic shape and format checking during export
- **Deployment**: Drop-in replacement for existing openpilot policy model

### Architecture Decisions
- **Recurrent**: GRU-based for temporal consistency (matches openpilot expectations)
- **MDN Outputs**: Mixture density networks for plan and curvature (probabilistic)
- **Interface**: Exact match to openpilot's policy input/output format
- **Domain Randomization**: Built-in for robust policy learning

## Testing Strategy

### Validation Levels
1. **Code Structure**: `python validate_code.py` (no dependencies)
2. **Component Testing**: `python test_pipeline.py` (mocked dependencies)
3. **Training Pipeline**: `python run_training.py --episodes 5` (full dependencies)
4. **Integration Testing**: Train → Export → Validate ONNX → Deploy

### Development Workflow
1. **Start Simple**: Use dummy environment for algorithm development
2. **Scale Up**: Move to CARLA for realistic scenarios
3. **Integrate Vision**: Add real openpilot vision model when available
4. **Deploy**: Export and test in openpilot replay tools

### Performance Validation
- **Lane Keeping**: Cross-track error < 0.5m (P95)
- **Comfort**: Jerk RMS < 2.0 m/s³
- **Safety**: Zero boundary violations (hard constraint)
- **Latency**: 20Hz operation on comma three hardware