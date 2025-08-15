# 🚀 Quick Start Guide

## Test Your Setup

```bash
# 1. Test if everything is ready
python test_training_ready.py

# 2. If dependencies are missing, install them
pip install torch numpy gymnasium onnx opencv-python matplotlib
```

## Run Training

```bash
# Simple training run (uses dummy environment, no CARLA needed)
python run_training.py --env dummy --episodes 50

# With your vision model files (if available)
python run_training.py \
    --env dummy \
    --vision-pkl driving_vision_dtr_tinygrad.pkl \
    --vision-meta driving_policy_dtr_metadata.pkl \
    --episodes 50
```

## What's Fixed

✅ **Import Path Issues**
- Openpilot modules now import correctly as `from selfdrive.modeld...`
- Added proper `__init__.py` to openpilot directory
- Path insertion happens before imports
- Graceful fallback when openpilot modules unavailable

✅ **Numpy Handling**
- Code works with or without numpy (falls back to pure Python)
- All numpy operations have fallback implementations
- Mock ModelConstants when openpilot unavailable

✅ **Logger Issues**
- Logger now initialized before use
- No more NameError on logger.warning

## Expected Output

When you run training, you should see:
```
=== Openpilot Policy Training Pipeline ===
=== Checking Dependencies ===
✓ torch: Available
✓ numpy: Available
✓ gymnasium: Available

=== Starting Training ===
Episode 1/50: return=-5.23, steps=156, policy_loss=0.4521, value_loss=0.2341
Episode 2/50: return=-4.87, steps=189, policy_loss=0.4102, value_loss=0.2156
...

=== Model Export Complete ===
Files created in ./output:
  - policy_model.pth (PyTorch checkpoint)
  - driving_policy.onnx (ONNX model for openpilot)
  - driving_policy_metadata.pkl (Metadata for openpilot)
```

## Troubleshooting

If you see import errors:
1. Make sure openpilot subdirectory exists
2. Check that openpilot/__init__.py was created
3. Verify numpy is installed: `pip install numpy`

If training fails:
1. Check GPU/CPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
2. Try smaller episodes: `--episodes 5`
3. Check logs for specific errors

## Next Steps

After successful training:
1. Check output directory for trained models
2. Validate ONNX export: `python -c "import onnx; onnx.checker.check_model('output/driving_policy.onnx')"`
3. Deploy to openpilot (see README.md for instructions)