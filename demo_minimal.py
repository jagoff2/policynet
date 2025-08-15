#!/usr/bin/env python3
"""
Minimal demo showing the policy training concept without heavy dependencies.
Shows the core architecture and training loop structure.
"""

import os
import sys
import random
import math

# Check if numpy is available, use basic Python if not
try:
    import numpy as np
    HAS_NUMPY = True
    print("Using NumPy for computations")
except ImportError:
    HAS_NUMPY = False
    print("Using pure Python for computations")
    
    # Minimal numpy-like functions
    class MockNumpy:
        @staticmethod
        def array(data):
            return data if isinstance(data, list) else [data]
        
        @staticmethod  
        def zeros(shape):
            if isinstance(shape, int):
                return [0.0] * shape
            elif len(shape) == 1:
                return [0.0] * shape[0]
            elif len(shape) == 2:
                return [[0.0] * shape[1] for _ in range(shape[0])]
            else:
                raise NotImplementedError("Only 1D and 2D supported")
        
        @staticmethod
        def random():
            return MockRandom()
    
    class MockRandom:
        @staticmethod
        def randn(*shape):
            if len(shape) == 1:
                return [random.gauss(0, 1) for _ in range(shape[0])]
            elif len(shape) == 2:
                return [[random.gauss(0, 1) for _ in range(shape[1])] for _ in range(shape[0])]
            else:
                return random.gauss(0, 1)
    
    np = MockNumpy()

def demo_policy_config():
    """Demonstrate the policy configuration concept."""
    print("\n=== Policy Configuration Demo ===")
    
    # Openpilot-compatible dimensions
    config = {
        'feature_len': 512,        # Vision features
        'history_len': 25,         # Input history buffer
        'desire_len': 8,           # Desire vector size
        'plan_mhp_n': 5,           # Plan mixture components
        'idx_n': 33,               # Plan time steps
        'plan_width': 15,          # Plan dimensions per timestep
    }
    
    print("PolicyConfig dimensions:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Calculate output dimensions
    plan_dim = config['idx_n'] * config['plan_width']  # 33 * 15 = 495
    plan_output_dim = config['plan_mhp_n'] * (2 * plan_dim + 1)  # 5 * (2*495 + 1) = 4955
    
    print(f"Computed dimensions:")
    print(f"  plan_dim: {plan_dim}")
    print(f"  plan_output_dim: {plan_output_dim}")
    print(f"  desired_curvature_output_dim: 2")  # Single mode: mean + log_std
    print(f"  desire_state_output_dim: 8")
    
    return config

def demo_environment():
    """Demonstrate the driving environment concept."""
    print("\n=== Dummy Driving Environment Demo ===")
    
    # Vehicle state: [lateral_offset, heading_error, velocity]
    state = [0.0, 0.0, 10.0]  # Start centered, straight, 10 m/s
    
    # Sinusoidal road parameters
    amplitude = 1.0
    frequency = 0.02
    phase = 0.0
    
    print("Initial state: [lateral_offset, heading_error, velocity]")
    print(f"  {state}")
    
    # Simulate a few steps
    dt = 0.05  # 20Hz
    time_steps = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    print("\nSimulation steps:")
    print("  Time | Action [curv, accel] | State [lat_off, head_err, vel] | Reward")
    
    for t in time_steps:
        # Simple control policy: follow road curvature
        road_curvature = 2 * math.pi * frequency * amplitude * math.cos(2 * math.pi * frequency * t)
        target_curvature = math.atan(road_curvature)
        
        # Action: [desired_curvature, desired_acceleration]
        action = [target_curvature * 0.5, 0.0]  # Gentle steering, maintain speed
        
        # Simple physics update
        curvature, accel = action
        lateral_offset, heading_error, velocity = state
        
        # Update state
        heading_error += curvature * velocity * dt
        lateral_offset += velocity * math.sin(heading_error) * dt
        velocity += accel * dt
        velocity = max(velocity, 0.1)  # Minimum speed
        
        state = [lateral_offset, heading_error, velocity]
        
        # Simple reward: penalize large lateral offset
        reward = -abs(lateral_offset) - 0.1 * abs(heading_error)
        
        print(f"  {t:4.1f} | [{action[0]:6.3f}, {action[1]:6.3f}] | [{state[0]:6.3f}, {state[1]:6.3f}, {state[2]:6.3f}] | {reward:6.3f}")

def demo_mdn_concept():
    """Demonstrate the MDN (Mixture Density Network) concept."""
    print("\n=== MDN Concept Demo ===")
    
    # MDN parameters for a 2-component, 1D mixture
    # Format: [mu1, mu2, log_std1, log_std2, logit1, logit2]
    mdn_params = [0.5, -0.3, -1.0, -0.5, 2.0, 1.0]
    
    print("MDN parameters (2 components, 1D):")
    print(f"  means: [{mdn_params[0]}, {mdn_params[1]}]")
    print(f"  log_stds: [{mdn_params[2]}, {mdn_params[3]}]")
    print(f"  logits: [{mdn_params[4]}, {mdn_params[5]}]")
    
    # Compute mixture weights (softmax of logits)
    logits = mdn_params[4:6]
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_logits)
    weights = [e / sum_exp for e in exp_logits]
    
    print(f"  weights: [{weights[0]:.3f}, {weights[1]:.3f}]")
    
    # Dominant component
    dominant_idx = 0 if weights[0] > weights[1] else 1
    dominant_mean = mdn_params[dominant_idx]
    
    print(f"  dominant component: {dominant_idx} (mean={dominant_mean:.3f})")
    
    return dominant_mean

def demo_training_loop():
    """Demonstrate the training loop concept."""
    print("\n=== Training Loop Demo ===")
    
    config = demo_policy_config()
    
    print("\nMini training simulation (5 episodes):")
    print("Episode | Steps | Return | Policy Loss | Value Loss")
    
    for episode in range(5):
        # Simulate episode
        episode_length = random.randint(50, 200)
        episode_return = random.uniform(-10.0, 5.0) + episode * 0.5  # Gradual improvement
        
        # Simulate losses (decreasing over time)
        policy_loss = random.uniform(0.1, 1.0) * math.exp(-episode * 0.1)
        value_loss = random.uniform(0.05, 0.5) * math.exp(-episode * 0.1)
        
        print(f"   {episode+1:2d}   | {episode_length:3d}   | {episode_return:6.2f} | {policy_loss:10.4f} | {value_loss:9.4f}")

def demo_onnx_export():
    """Demonstrate the ONNX export concept."""
    print("\n=== ONNX Export Demo ===")
    
    config = demo_policy_config()
    
    # Simulated model outputs
    outputs = {
        'plan': f"tensor shape: (1, {config['plan_mhp_n'] * (2 * config['idx_n'] * config['plan_width'] + 1)})",
        'desired_curvature': "tensor shape: (1, 2)",  # mean + log_std
        'desire_state': f"tensor shape: (1, {config['desire_len']})"
    }
    
    print("Simulated policy outputs for ONNX export:")
    for name, shape in outputs.items():
        print(f"  {name}: {shape}")
    
    # Metadata that would be generated
    metadata = {
        'input_shapes': {
            'features_buffer': f"(1, {config['history_len']}, {config['feature_len']})",
            'desire': f"(1, {config['history_len']}, {config['desire_len']})",
            'traffic_convention': "(1, 2)",
            'lateral_control_params': "(1, 2)",
            'prev_desired_curv': f"(1, {config['history_len']}, 1)"
        },
        'output_slices': {
            'plan': f"slice(0, {config['plan_mhp_n'] * (2 * config['idx_n'] * config['plan_width'] + 1)})",
            'desired_curvature': f"slice(..., ...)",
            'desire_state': f"slice(..., {config['desire_len']})"
        }
    }
    
    print("\nGenerated metadata:")
    print("  Input shapes:")
    for name, shape in metadata['input_shapes'].items():
        print(f"    {name}: {shape}")
    
    print("  Output slices ready for openpilot parser")

def main():
    """Run the complete demo."""
    print("=== PolicyNet Architecture Demo ===")
    print("Demonstrating core concepts without heavy dependencies")
    
    demo_policy_config()
    demo_environment() 
    demo_mdn_concept()
    demo_training_loop()
    demo_onnx_export()
    
    print("\n=== Demo Complete ===")
    print("This shows the conceptual structure of the training system.")
    print("\nTo run actual training:")
    print("1. Install dependencies: python install_deps.py")
    print("2. Run training: python run_training.py --env dummy --episodes 10")
    print("3. Deploy to openpilot: copy output files to openpilot/models/")

if __name__ == "__main__":
    main()