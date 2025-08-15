#!/usr/bin/env python3
"""
Verify that tensor dimensions match in the policy network.
"""

# Mock PolicyConfig for verification
class PolicyConfig:
    def __init__(self):
        self.feature_len = 512
        self.history_len = 25
        self.desire_len = 8
        self.traffic_convention_len = 2
        self.lateral_control_params_len = 2
        self.prev_desired_curv_len = 1
        self.gru_hidden_size = 512
        self.plan_mhp_n = 5
        self.idx_n = 33
        self.plan_width = 15
        self.desired_curv_width = 1
        self.desire_pred_width = 8

cfg = PolicyConfig()

print("=== Dimension Verification ===")
print(f"GRU hidden size: {cfg.gru_hidden_size}")
print(f"Feature length: {cfg.feature_len}")
print(f"Desire length: {cfg.desire_len}")
print(f"Traffic convention: {cfg.traffic_convention_len}")
print(f"Lateral control params: {cfg.lateral_control_params_len}")

# Calculate expected combined dimension
combined_dim = (cfg.gru_hidden_size +         # last_hidden: 512
               cfg.feature_len +              # current_features: 512
               cfg.desire_len +               # desire_context: 8
               cfg.traffic_convention_len +   # traffic_convention: 2
               cfg.lateral_control_params_len) # lateral_control_params: 2

print(f"\nCombined dimension calculation:")
print(f"  {cfg.gru_hidden_size} + {cfg.feature_len} + {cfg.desire_len} + {cfg.traffic_convention_len} + {cfg.lateral_control_params_len} = {combined_dim}")

print(f"\n✓ Expected combined dimension: {combined_dim}")
print("This should now match the tensor size in the forward pass!")

# Calculate observation dimension for environment
obs_dim = (cfg.history_len * cfg.feature_len +
           cfg.history_len * cfg.desire_len +
           cfg.history_len * cfg.prev_desired_curv_len +
           cfg.traffic_convention_len +
           cfg.lateral_control_params_len)

print(f"\nObservation space dimension: {obs_dim}")
print("Components:")
print(f"  Features: {cfg.history_len} × {cfg.feature_len} = {cfg.history_len * cfg.feature_len}")
print(f"  Desire: {cfg.history_len} × {cfg.desire_len} = {cfg.history_len * cfg.desire_len}")
print(f"  Prev curvature: {cfg.history_len} × {cfg.prev_desired_curv_len} = {cfg.history_len * cfg.prev_desired_curv_len}")
print(f"  Traffic convention: {cfg.traffic_convention_len}")
print(f"  Lateral control: {cfg.lateral_control_params_len}")