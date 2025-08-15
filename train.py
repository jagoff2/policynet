"""
train.py
========

Entry point for training an openpilot-compatible policy model using
reinforcement learning. This script constructs the environment, policy
network and value function, then runs a clipped PPO optimisation loop.
The resulting policy can be exported to ONNX and deployed alongside
comma.ai's vision model.

The training pipeline supports both a full CARLA-based environment and
a lightweight dummy environment for rapid iteration. The dummy
environment allows basic testing of the RL code without requiring the
heavy CARLA simulator. The environment type can be selected via the
command line.

Usage::

    python -m carla_policy_training.train \
        --vision-onnx /path/to/driving_vision.onnx \
        --vision-meta /path/to/driving_vision_metadata.pkl \
        --env dummy \
        --output-dir /tmp/rl_model

Dependencies:

* PyTorch (1.10+)
* numpy
* gym (0.26+) for the dummy environment
* CARLA 0.10.0 (optional, for the carla environment)
* openpilot and tinygrad (optional, for using the real vision model)

"""

from __future__ import annotations

import argparse
import os
import math
import time
import logging
from typing import Dict, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
except ImportError as e:
    raise ImportError("PyTorch must be installed to run training: {}".format(e))

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore

from mdn import mdn_split_params, mdn_log_prob, mdn_mean, safe_softmax
from policy_model import PolicyConfig, PolicyNetwork, export_to_onnx, save_metadata
from openpilot_wrapper import OpenPilotWrapper
from carla_env import DummyDrivingEnv, CarlaDrivingEnv


class ValueNetwork(nn.Module):
    """Simple MLP value function approximator for PPO.

    Accepts a flattened observation vector and outputs a scalar value.
    The architecture comprises two hidden layers with ReLU activations.
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def unpack_observation(obs: np.ndarray, cfg: PolicyConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a flattened observation into constituent tensors for the policy.

    Args:
        obs: Flattened observation array of shape (obs_dim,).
        cfg: PolicyConfig describing the sizes of each component.

    Returns:
        Tuple of tensors (features_buffer, desire, traffic_convention,
        lateral_control_params, prev_curv) ready for input into the
        policy network. All tensors are of shape (1, ...).
    """
    offset = 0
    # features_buffer: (H, F)
    feat_size = cfg.history_len * cfg.feature_len
    feat = obs[offset : offset + feat_size].reshape(cfg.history_len, cfg.feature_len)
    offset += feat_size
    # desire: (H, D)
    des_size = cfg.history_len * cfg.desire_len
    des = obs[offset : offset + des_size].reshape(cfg.history_len, cfg.desire_len)
    offset += des_size
    # prev_desired_curv: (H, C)
    prev_size = cfg.history_len * cfg.prev_desired_curv_len
    prev = obs[offset : offset + prev_size].reshape(cfg.history_len, cfg.prev_desired_curv_len)
    offset += prev_size
    # traffic_convention: (2,)
    traffic = obs[offset : offset + cfg.traffic_convention_len]
    offset += cfg.traffic_convention_len
    # lateral_control_params: (2,)
    lat = obs[offset : offset + cfg.lateral_control_params_len]
    # Convert to tensors with batch dimension
    return (
        torch.from_numpy(feat).unsqueeze(0),
        torch.from_numpy(des).unsqueeze(0),
        torch.from_numpy(traffic).unsqueeze(0),
        torch.from_numpy(lat).unsqueeze(0),
        torch.from_numpy(prev).unsqueeze(0),
    )


def extract_action_and_log_prob(
    outputs: Dict[str, torch.Tensor], cfg: PolicyConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute continuous action and log probability from policy outputs.

    The action consists of two components: desired curvature and desired
    acceleration. The curvature is taken as the mixture mean of the
    ``desired_curvature`` MDN head. The acceleration is derived from the
    first time step of the plan's acceleration column. Both are
    concatenated into a single tensor of shape (batch, 2).

    The log probability of the action under a combined two-dimensional
    mixture distribution is computed by fusing the plan and curvature
    MDN parameters. This allows PPO to measure how likely the chosen
    action is relative to the current policy. Note that the plan MDN
    contains many more dimensions than just acceleration; only the
    acceleration dimension is extracted.

    Args:
        outputs: Dictionary of raw policy outputs from ``PolicyNetwork``.
        cfg: PolicyConfig instance.

    Returns:
        Tuple (action, log_prob) where ``action`` is a tensor of shape
        (batch, 2) containing curvature and acceleration, and ``log_prob``
        is a tensor of shape (batch,) containing the log probability.
    """
    # Unpack outputs
    plan_raw = outputs["plan"]  # shape (B, plan_output_dim)
    curv_raw = outputs["desired_curvature"]  # shape (B, desired_curv_output_dim)
    # Parse desired curvature MDN
    mu_c, log_std_c, logits_c = mdn_split_params(curv_raw, n_mix=1)
    mu_c = mu_c.squeeze(-2)  # (B, 1)
    log_std_c = log_std_c.squeeze(-2)  # (B, 1)
    curv_action = mu_c.squeeze(-1)  # (B,)
    # Parse plan MDN
    mu_p, log_std_p, logits_p = mdn_split_params(plan_raw, n_mix=cfg.plan_mhp_n)
    # Compute mixture weights
    weights_p = safe_softmax(logits_p, dim=-1)  # (B, n_mix)
    # Compute weighted mixture mean for plan
    # mu_p: (B, n_mix, plan_dim)
    weighted_mu_p = (weights_p.unsqueeze(-1) * mu_p).sum(dim=-2)  # (B, plan_dim)
    # Reshape to (B, idx_n, plan_width)
    plan_mean = weighted_mu_p.view(-1, cfg.idx_n, cfg.plan_width)
    # Acceleration dimension assumed to be the last column in plan_width
    # See openpilot Plan definitions for exact index; adjust if necessary
    accel_action = plan_mean[:, 0, -1]  # (B,)
    # Compute log probability of (curv, accel) under combined mixture
    # Expand curvature MDN params to match plan mixture components
    # mu_c/log_std_c: (B, 1); replicate to (B, n_mix)
    mu_c_rep = mu_c.expand(-1, cfg.plan_mhp_n)
    log_std_c_rep = log_std_c.expand(-1, cfg.plan_mhp_n)
    # Extract acceleration mean and std per mixture
    # Plan.ACCELERATION corresponds to indices 6:9 in each plan time step.
    # We take the first component (x-axis acceleration) as the control
    plan_dim = cfg.plan_dim
    # Calculate index of acceleration x component in the flattened plan vector
    # Each time step has plan_width elements; acceleration starts at index 6
    accel_offset = cfg.idx_n * 0 + 6  # first timestep, start slice index
    plan_accel_mu = mu_p[..., accel_offset]
    plan_accel_log_std = log_std_p[..., accel_offset]
    # Combine curvature and acceleration into 2D mixture parameters
    # combined_mu: (B, n_mix, 2)
    combined_mu = torch.stack([mu_c_rep, plan_accel_mu], dim=-1)
    combined_log_std = torch.stack([log_std_c_rep, plan_accel_log_std], dim=-1)
    
    # Format for mdn_log_prob: [all_mus, all_log_stds, all_logits]
    # Shape: (B, n_mix * (2*d + 1)) where d=2 for 2D action
    logits_comb = logits_p  # (B, n_mix)
    
    # Flatten means: (B, n_mix, 2) -> (B, n_mix*2)
    flat_mu = combined_mu.view(combined_mu.shape[0], -1)
    # Flatten log_stds: (B, n_mix, 2) -> (B, n_mix*2) 
    flat_log_std = combined_log_std.view(combined_log_std.shape[0], -1)
    
    # Concatenate in expected order: [means, log_stds, logits]
    flat_params = torch.cat([flat_mu, flat_log_std, logits_comb], dim=-1)  # (B, n_mix*(2+2+1))
    # Form action tensor
    action = torch.stack([curv_action, accel_action], dim=-1)  # (B, 2)
    # Compute log_prob using mdn_log_prob util (over 2D)
    log_prob = mdn_log_prob(action, flat_params, n_mix=cfg.plan_mhp_n)  # (B,)
    return action, log_prob


def compute_returns_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute discounted returns and GAE advantages.

    Args:
        rewards: Array of rewards of shape (T,).
        values: Array of value estimates of shape (T+1,).
        dones: Binary array indicating episode termination of shape (T,).
        gamma: Discount factor.
        lam: GAE lambda parameter.

    Returns:
        Tuple (returns, advantages) both of shape (T,).
    """
    T = len(rewards)
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0.0
    last_return = values[-1]
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * lam * nonterminal * last_gae_lam
        returns[t] = last_return = rewards[t] + gamma * last_return * nonterminal
    return returns, advantages


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an openpilot-compatible policy with RL")
    parser.add_argument("--vision-onnx", type=str, required=False, default="driving_vision.onnx", help="Path to vision model ONNX file")
    parser.add_argument("--vision-meta", type=str, required=False, default="driving_vision_metadata.pkl", help="Path to vision metadata pickle")
    parser.add_argument("--env", type=str, choices=["dummy", "carla"], default="dummy", help="Environment type")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--timesteps", type=int, default=500, help="Maximum timesteps per episode")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save model and logs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ppo-clip", type=float, default=0.2, help="PPO clipping range")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of steps to collect before each update")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Initialise openpilot wrapper (dummy fallback if unavailable)
    op_wrapper = OpenPilotWrapper(args.vision_onnx, args.vision_meta)
    # Create policy config matching openpilot exactly
    cfg = PolicyConfig()
    
    # Verify feature length matches
    if op_wrapper.feature_len != cfg.feature_len:
        logger.warning(f"Feature length mismatch: wrapper={op_wrapper.feature_len}, config={cfg.feature_len}")
        cfg.feature_len = op_wrapper.feature_len
    # Select environment
    if args.env == "carla":
        env = CarlaDrivingEnv(op_wrapper, cfg, seed=args.seed)
    else:
        env = DummyDrivingEnv(op_wrapper, cfg, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    # Instantiate networks
    policy = PolicyNetwork(cfg)
    value_fn = ValueNetwork(obs_dim)
    policy.to(torch.device("cpu"))
    value_fn.to(torch.device("cpu"))
    # Optimizers
    optimizer_p = optim.Adam(policy.parameters(), lr=args.lr)
    optimizer_v = optim.Adam(value_fn.parameters(), lr=args.lr)
    # Initialize openpilot parser for deployment-compatible outputs
    parser = None
    if hasattr(op_wrapper, 'parser') and op_wrapper.parser is not None:
        parser = op_wrapper.parser
        logger.info("Using parser from openpilot wrapper")
    else:
        try:
            # Try to initialize parser directly
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            openpilot_path = os.path.join(current_dir, 'openpilot')
            
            if os.path.exists(openpilot_path) and openpilot_path not in sys.path:
                sys.path.insert(0, openpilot_path)
                logger.info(f"Added openpilot path for parser: {openpilot_path}")
            
            from selfdrive.modeld.parse_model_outputs import Parser
            parser = Parser()
            logger.info("Initialized openpilot parser successfully")
        except Exception as e:
            logger.warning(f"Could not initialize parser: {e}")
            parser = None
    # Main training loop
    global_step = 0
    for episode in range(args.episodes):
        obs = env.reset()
        policy.reset_state(batch_size=1)
        done = False
        ep_rewards = []
        ep_log_probs = []
        ep_values = []
        ep_obs = []
        ep_actions = []
        for t in range(args.timesteps):
            # Convert observation to policy inputs
            (feat_t, des_t, tc_t, lat_t, prev_t) = unpack_observation(obs, cfg)
            
            # Policy forward pass
            with torch.no_grad():
                outputs = policy(feat_t.float(), des_t.float(), tc_t.float(), lat_t.float(), prev_t.float())
            
            # Extract action and log probability for PPO
            action_t, log_prob_t = extract_action_and_log_prob(outputs, cfg)
            
            # Optional: validate against openpilot parser for deployment compatibility
            if parser is not None and episode == 0 and t < 5:  # Only validate first few steps
                try:
                    raw_out = {
                        "plan": outputs["plan"].detach().numpy(),
                        "desired_curvature": outputs["desired_curvature"].detach().numpy(), 
                        "desire_state": outputs["desire_state"].detach().numpy(),
                    }
                    parsed = parser.parse_policy_outputs(raw_out)
                    logger.info(f"Parser validation successful at step {t}")
                except Exception as e:
                    logger.warning(f"Parser validation failed: {e}")
            # Value estimate
            value_t = value_fn(torch.from_numpy(obs).float().unsqueeze(0))
            # Execute action
            next_obs, reward, done, _ = env.step(action_t.detach().numpy().squeeze())
            # Store
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob_t.detach().item())
            ep_values.append(value_t.detach().item())
            ep_obs.append(obs)
            ep_actions.append(action_t.detach().numpy().squeeze())
            obs = next_obs
            global_step += 1
            if done:
                break
        # Add final value
        with torch.no_grad():
            last_value = value_fn(torch.from_numpy(obs).float().unsqueeze(0)).item() if not done else 0.0
        ep_values.append(last_value)
        # Convert to arrays
        rewards_np = np.array(ep_rewards, dtype=np.float32)
        values_np = np.array(ep_values, dtype=np.float32)
        dones_np = np.array([0.0 if i < len(ep_rewards) - 1 else float(done) for i in range(len(ep_rewards))], dtype=np.float32)
        # Compute returns and advantages
        returns, advantages = compute_returns_advantages(rewards_np, values_np, dones_np, args.gamma, args.gae_lambda)
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Convert to tensors
        obs_tensor = torch.from_numpy(np.stack(ep_obs)).float()
        actions_tensor = torch.from_numpy(np.stack(ep_actions)).float()
        log_probs_old = torch.from_numpy(np.array(ep_log_probs)).float()
        returns_tensor = torch.from_numpy(returns).float()
        advantages_tensor = torch.from_numpy(advantages).float()
        # Update policy and value via PPO
        policy.reset_state(batch_size=len(ep_obs))
        value_fn.train()
        policy.train()
        # Batch update on collected trajectory
        for _ in range(4):  # number of epochs per update
            # Since our data is small, we can process all at once
            total_loss_p = 0.0
            total_loss_v = 0.0
            # Forward policy for each step individually to maintain correct RNN state
            policy_outputs = {"plan": [], "desired_curvature": [], "desire_state": []}
            # Reset hidden state for sequence
            policy.reset_state(batch_size=1)
            for i in range(len(ep_obs)):
                (feat_i, des_i, tc_i, lat_i, prev_i) = unpack_observation(ep_obs[i], cfg)
                out = policy(
                    feat_i.float(), des_i.float(), tc_i.float(), lat_i.float(), prev_i.float()
                )
                policy_outputs["plan"].append(out["plan"])
                policy_outputs["desired_curvature"].append(out["desired_curvature"])
                policy_outputs["desire_state"].append(out["desire_state"])
            # Stack outputs
            for k in policy_outputs:
                policy_outputs[k] = torch.cat(policy_outputs[k], dim=0)
            # Compute new actions and log probs
            new_actions, new_log_probs = extract_action_and_log_prob(policy_outputs, cfg)
            # Ratio for PPO
            ratio = torch.exp(new_log_probs - log_probs_old)
            # Policy loss
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * advantages_tensor
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            # Entropy bonus for exploration - use curvature log_std
            curv_raw = policy_outputs["desired_curvature"]
            if curv_raw.shape[-1] >= 2:  # Has log_std
                curv_log_std = curv_raw[:, 1]
                entropy = 0.5 * (1 + torch.log(torch.tensor(2 * math.pi)) + 2 * curv_log_std)
                policy_loss -= 0.001 * entropy.mean()
            # Value loss
            values_pred = value_fn(obs_tensor)
            value_loss = 0.5 * (returns_tensor - values_pred).pow(2).mean()
            # Backprop
            optimizer_p.zero_grad()
            optimizer_v.zero_grad()
            total_loss = policy_loss + value_loss
            total_loss.backward()
            optimizer_p.step()
            optimizer_v.step()
            total_loss_p += policy_loss.item()
            total_loss_v += value_loss.item()
        avg_reward = rewards_np.sum()
        print(
            f"Episode {episode+1}/{args.episodes}: return={avg_reward:.2f}, steps={len(ep_rewards)}, "
            f"policy_loss={total_loss_p:.4f}, value_loss={total_loss_v:.4f}"
        )
        
        # Log additional metrics and early stopping
        if episode == 0 or (episode + 1) % 5 == 0:
            with torch.no_grad():
                # Sample from current policy for analysis
                test_obs = ep_obs[0] if ep_obs else obs
                test_feat, test_des, test_tc, test_lat, test_prev = unpack_observation(test_obs, cfg)
                test_outputs = policy(test_feat.float(), test_des.float(), test_tc.float(), test_lat.float(), test_prev.float())
                test_action, _ = extract_action_and_log_prob(test_outputs, cfg)
                logger.info(f"Episode {episode+1} policy sample: curvature={test_action[0][0]:.3f}, accel={test_action[0][1]:.3f}")
        
        # Early stopping for reasonable performance
        if avg_reward > -5.0 and episode > 10:
            logger.info(f"Early stopping at episode {episode+1} - performance threshold reached")
            break
    # Save trained model and create openpilot-compatible exports
    print("\n=== Exporting trained model ===")
    
    # Save PyTorch model
    model_path = os.path.join(args.output_dir, "policy_model.pth")
    torch.save({
        'model_state_dict': policy.state_dict(),
        'config': cfg,
        'training_args': vars(args)
    }, model_path)
    print(f"Saved PyTorch model to {model_path}")
    
    # Export to ONNX with proper validation
    onnx_path = os.path.join(args.output_dir, "driving_policy.onnx")
    metadata_path = os.path.join(args.output_dir, "driving_policy_metadata.pkl")
    
    # Create example inputs matching openpilot format
    policy.eval()
    policy.reset_state(batch_size=1)
    
    dummy_inputs = (
        torch.zeros(1, cfg.history_len, cfg.feature_len),  # features_buffer
        torch.zeros(1, cfg.history_len, cfg.desire_len),   # desire
        torch.zeros(1, cfg.traffic_convention_len),        # traffic_convention
        torch.zeros(1, cfg.lateral_control_params_len),    # lateral_control_params
        torch.zeros(1, cfg.history_len, cfg.prev_desired_curv_len)  # prev_desired_curv
    )
    
    # Export ONNX model
    export_to_onnx(policy, dummy_inputs, onnx_path)
    
    # Save metadata
    save_metadata(cfg, metadata_path)
    
    print(f"\n=== Model Export Complete ===")
    print(f"Files created in {args.output_dir}:")
    print(f"  - policy_model.pth (PyTorch checkpoint)")
    print(f"  - driving_policy.onnx (ONNX model for openpilot)")
    print(f"  - driving_policy_metadata.pkl (Metadata for openpilot)")
    print(f"\nTo deploy in openpilot:")
    print(f"  1. Copy ONNX and metadata files to openpilot model directory")
    print(f"  2. Update model loading code to use the new policy")
    print(f"  3. Test with openpilot's replay tools")


if __name__ == "__main__":
    main()
