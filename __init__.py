"""
carla_policy_training
=====================

This package contains modules to train an openpilot-compatible policy model
using reinforcement learning inside the CARLA simulator. The training pipeline
freezes the existing comma.ai vision model and learns a new policy network
from scratch. The policy network ingests a history of vision features,
desire pulses, traffic convention flags and latency parameters and outputs
a mixture density trajectory plan, desired curvature and desire state
probabilities. The resulting model can be exported to ONNX and loaded
directly by openpilot's modeld process.

Key modules:

- ``mdn.py``: mixture density network utilities for packing and unpacking
  Gaussian mixtures, computing likelihoods and sampling.
- ``policy_model.py``: defines the recurrent policy network using PyTorch.
- ``openpilot_wrapper.py``: helpers to load the comma vision model via
  openpilot's tinygrad runner and extract hidden state features. Also
  contains parsing utilities mirroring openpilot's parser.
- ``carla_env.py``: a Gym-compatible environment for driving in CARLA,
  randomising road layouts and injecting vision-derived observations.
- ``train.py``: main script to configure the environment and policy,
  run a recurrent PPO algorithm and export the trained model.

Note that this package depends on external libraries such as PyTorch and
CARLA. To execute training you must have these installed along with
openpilot and its dependencies. See the README for installation details.
"""
