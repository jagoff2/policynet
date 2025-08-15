"""
policy_model.py
================

Defines the recurrent policy network that produces openpilot-compatible
outputs from a history of vision embeddings and auxiliary inputs. The
network is built with PyTorch and outputs mixture density parameters
for a trajectory plan, a desired curvature value and desire state
probabilities. It maintains a recurrent hidden state internally and
exposes methods for both forward inference and state resetting.

The network is designed to be flexible: the number of mixture
components, the dimensionality of the plan and curvature outputs, and
the history length can all be configured via the ``PolicyConfig``
dataclass.

Note: this module depends on PyTorch; install it before running the
training or inference scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import pickle

import torch
import torch.nn as nn

from mdn import mdn_split_params


@dataclass
class PolicyConfig:
    """Configuration options for the policy network.

    All dimensions match openpilot ModelConstants exactly for seamless
    drop-in compatibility.
    """
    
    # From ModelConstants - these are fixed for openpilot compatibility
    feature_len: int = 512  # ModelConstants.FEATURE_LEN
    history_len: int = 25   # ModelConstants.INPUT_HISTORY_BUFFER_LEN
    desire_len: int = 8     # ModelConstants.DESIRE_LEN
    traffic_convention_len: int = 2  # ModelConstants.TRAFFIC_CONVENTION_LEN
    lateral_control_params_len: int = 2  # ModelConstants.LATERAL_CONTROL_PARAMS_LEN
    prev_desired_curv_len: int = 1  # ModelConstants.PREV_DESIRED_CURV_LEN
    idx_n: int = 33  # ModelConstants.IDX_N
    plan_width: int = 15  # ModelConstants.PLAN_WIDTH
    desired_curv_width: int = 1  # ModelConstants.DESIRED_CURV_WIDTH
    desire_pred_width: int = 8  # ModelConstants.DESIRE_PRED_WIDTH
    plan_mhp_n: int = 5  # ModelConstants.PLAN_MHP_N
    plan_mhp_selection: int = 1  # ModelConstants.PLAN_MHP_SELECTION
    
    # Architecture hyperparameters
    gru_hidden_size: int = 512
    mixture_log_sigma_clip: float = 5.0

    @property
    def plan_dim(self) -> int:
        """Total dimensionality of the plan target (idx_n * plan_width)."""
        return self.idx_n * self.plan_width

    @property
    def plan_output_dim(self) -> int:
        """Size of the flattened MDN parameters for plan output.
        
        For N mixtures over D-dimensional output:
        Each mixture needs D means + D log_stds + 1 logit = 2*D + 1
        Total: N * (2*D + 1)
        """
        return self.plan_mhp_n * (2 * self.plan_dim + 1)

    @property
    def desired_curv_output_dim(self) -> int:
        """Size of the flattened MDN parameters for desired curvature."""
        # Single mode MDN: mean + log_std (no mixture logits for single mode)
        return 2 * self.desired_curv_width


class PolicyNetwork(nn.Module):
    """Recurrent policy network producing openpilot-compatible outputs.

    The network consumes a history of vision features along with the current
    desire pulse, traffic convention flags and latency parameters. It
    maintains an internal GRU state across timesteps. For each call, it
    returns a dictionary of raw MDN parameters and desire state logits.

    Example usage::

        cfg = PolicyConfig(feature_len=512, history_len=2,
                           desire_len=8, idx_n=33, plan_width=6)
        policy = PolicyNetwork(cfg)
        # Reset hidden state at the start of an episode
        policy.reset_state(batch_size=1)
        # Forward pass with input tensors
        outputs = policy(features_buffer, desire, traffic_convention,
                         lateral_control_params, prev_desired_curv)
        plan_params = outputs['plan']  # shape (batch, plan_output_dim)
        curvature_params = outputs['desired_curvature']
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        
        # Layer norm for input normalization
        self.input_norm = nn.LayerNorm(cfg.feature_len + cfg.desire_len + cfg.prev_desired_curv_len)
        
        # Input projection for GRU
        input_dim = cfg.feature_len + cfg.desire_len + cfg.prev_desired_curv_len
        self.input_proj = nn.Linear(input_dim, cfg.gru_hidden_size)
        
        # GRU over temporal dimension (recurrent backbone)
        self.gru = nn.GRU(cfg.gru_hidden_size, cfg.gru_hidden_size, batch_first=True)
        
        # Post-GRU projection combining GRU output with current features and side inputs
        side_dim = cfg.traffic_convention_len + cfg.lateral_control_params_len
        # Total: gru_hidden + current_features + desire_context + side_inputs
        combined_dim = cfg.gru_hidden_size + cfg.feature_len + cfg.desire_len + side_dim
        self.post_proj = nn.Linear(combined_dim, cfg.gru_hidden_size)
        
        # Output heads
        self.plan_head = nn.Linear(cfg.gru_hidden_size, cfg.plan_output_dim)
        self.curv_head = nn.Linear(cfg.gru_hidden_size, cfg.desired_curv_output_dim)
        self.desire_state_head = nn.Linear(cfg.gru_hidden_size, cfg.desire_pred_width)
        
        # Initialize GRU hidden state buffer
        self.register_buffer("_h0", torch.zeros(1, 1, cfg.gru_hidden_size))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize linear layers
        for module in [self.input_proj, self.post_proj, self.plan_head, self.curv_head, self.desire_state_head]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset the recurrent hidden state.

        Should be called at the beginning of each new episode.

        Args:
            batch_size: Number of parallel environments.
        """
        # Initialize hidden state for the batch
        device = next(self.parameters()).device
        self.h = self._h0.expand(1, batch_size, -1).contiguous().to(device)

    def forward(
        self,
        features_buffer: torch.Tensor,
        desire: torch.Tensor,
        traffic_convention: torch.Tensor,
        lateral_control_params: torch.Tensor,
        prev_desired_curv: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass matching openpilot's expected input/output format.

        Args:
            features_buffer: Vision features (B, H, F) where H=25, F=512
            desire: Desire pulses (B, H, D) where D=8 - rising edge only
            traffic_convention: (B, 2) - one-hot [LHD, RHD]
            lateral_control_params: (B, 2) - [v_ego, lat_delay]
            prev_desired_curv: (B, H, 1) - previous curvature commands

        Returns:
            Dict with openpilot-compatible outputs:
            - 'plan': MDN parameters for trajectory plan
            - 'desired_curvature': MDN parameters for desired curvature  
            - 'desire_state': softmax logits for desire state prediction
        """
        B, H, F = features_buffer.shape
        
        # Combine temporal inputs for each timestep
        step_input = torch.cat([
            features_buffer,  # (B, H, 512)
            desire,          # (B, H, 8) 
            prev_desired_curv,  # (B, H, 1)
        ], dim=-1)  # (B, H, 521)
        
        # Apply layer normalization and input projection
        step_input = self.input_norm(step_input)
        step_proj = torch.tanh(self.input_proj(step_input))  # (B, H, hidden_size)
        
        # Recurrent processing over time
        gru_out, self.h = self.gru(step_proj, self.h)  # (B, H, hidden_size)
        
        # Extract final hidden state and combine with current context
        last_hidden = gru_out[:, -1, :]  # (B, hidden_size)
        current_features = features_buffer[:, -1, :]  # (B, 512) - most recent features
        
        # Aggregate max-pooled desire over history (for latched behavior)
        desire_context = torch.max(desire, dim=1)[0]  # (B, 8)
        
        # Debug: print tensor shapes to identify mismatch
        # print(f"Debug shapes: last_hidden={last_hidden.shape}, current_features={current_features.shape}")
        # print(f"  desire_context={desire_context.shape}, traffic_convention={traffic_convention.shape}")
        # print(f"  lateral_control_params={lateral_control_params.shape}")
        
        # Combine all context for final processing
        combined_context = torch.cat([
            last_hidden,               # (B, 512)
            current_features,          # (B, 512)  
            desire_context,            # (B, 8)
            traffic_convention,        # (B, 2)
            lateral_control_params,    # (B, 2)
        ], dim=-1)  # Expected total: 512 + 512 + 8 + 2 + 2 = 1036
        
        # Debug: check combined context size
        # print(f"Combined context shape: {combined_context.shape}, expected: (B, 1036)")
        
        # Final processing layer
        x = torch.relu(self.post_proj(combined_context))  # (B, 512)
        
        # Generate outputs
        plan_raw = self.plan_head(x)  # (B, plan_output_dim)
        curv_raw = self.curv_head(x)  # (B, desired_curv_output_dim) 
        desire_state_logits = self.desire_state_head(x)  # (B, 8)
        
        # Apply log-std clipping for numerical stability
        curv_raw = self._clip_mdn_log_std(curv_raw, single_mode=True)
        plan_raw = self._clip_mdn_log_std(plan_raw, single_mode=False)
        
        return {
            "plan": plan_raw,
            "desired_curvature": curv_raw,
            "desire_state": desire_state_logits,
        }
    
    def _clip_mdn_log_std(self, params: torch.Tensor, single_mode: bool = False) -> torch.Tensor:
        """Clip log-std parameters to avoid numerical issues."""
        if single_mode:
            # For single-mode MDN: [mean, log_std]
            mean, log_std = params.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, max=self.cfg.mixture_log_sigma_clip)
            return torch.cat([mean, log_std], dim=-1)
        else:
            # For multi-mode MDN: [means, log_stds, logits] per mixture
            # Reshape to (B, n_mix, 2*dim + 1)
            B = params.shape[0]
            params = params.view(B, self.cfg.plan_mhp_n, -1)
            dim_size = (params.shape[-1] - 1) // 2
            
            means = params[..., :dim_size]
            log_stds = params[..., dim_size:2*dim_size] 
            logits = params[..., -1:]
            
            # Clip log stds
            log_stds = torch.clamp(log_stds, max=self.cfg.mixture_log_sigma_clip)
            
            # Recombine and flatten
            clipped = torch.cat([means, log_stds, logits], dim=-1)
            return clipped.view(B, -1)


def export_to_onnx(model: PolicyNetwork, example_inputs: Tuple[torch.Tensor, ...], output_path: str) -> None:
    """Export policy network to ONNX with openpilot-compatible metadata.
    
    The exported model will have the exact input/output format expected by
    openpilot's tinygrad runtime and parser.
    """
    import torch
    
    model.eval()
    
    # Ensure model is in inference mode
    with torch.no_grad():
        # Test forward pass to validate
        test_outputs = model(*example_inputs)
        
        # Verify output shapes match expected openpilot format
        expected_plan_dim = model.cfg.plan_mhp_n * (2 * model.cfg.plan_dim + 1)
        expected_curv_dim = 2 * model.cfg.desired_curv_width
        expected_desire_dim = model.cfg.desire_pred_width
        
        assert test_outputs["plan"].shape[-1] == expected_plan_dim, f"Plan output shape mismatch: {test_outputs['plan'].shape[-1]} vs {expected_plan_dim}"
        assert test_outputs["desired_curvature"].shape[-1] == expected_curv_dim, f"Curvature output shape mismatch: {test_outputs['desired_curvature'].shape[-1]} vs {expected_curv_dim}"
        assert test_outputs["desire_state"].shape[-1] == expected_desire_dim, f"Desire state shape mismatch: {test_outputs['desire_state'].shape[-1]} vs {expected_desire_dim}"
        
        print(f"✓ Output validation passed:")
        print(f"  Plan: {test_outputs['plan'].shape} -> {expected_plan_dim}")
        print(f"  Curvature: {test_outputs['desired_curvature'].shape} -> {expected_curv_dim}")
        print(f"  Desire state: {test_outputs['desire_state'].shape} -> {expected_desire_dim}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        example_inputs,
        output_path,
        input_names=[
            "features_buffer",      # (B, 25, 512)
            "desire",              # (B, 25, 8)
            "traffic_convention",  # (B, 2)
            "lateral_control_params", # (B, 2)
            "prev_desired_curv",   # (B, 25, 1)
        ],
        output_names=[
            "plan",               # (B, plan_output_dim)
            "desired_curvature",  # (B, desired_curv_output_dim)
            "desire_state",       # (B, 8)
        ],
        opset_version=15,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        export_params=True,
        dynamic_axes={
            "features_buffer": {0: "batch"},
            "desire": {0: "batch"},
            "traffic_convention": {0: "batch"},
            "lateral_control_params": {0: "batch"},
            "prev_desired_curv": {0: "batch"},
            "plan": {0: "batch"},
            "desired_curvature": {0: "batch"},
            "desire_state": {0: "batch"},
        },
    )
    
    print(f"✓ ONNX model exported to {output_path}")
    
    # Validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model validation passed")
    except ImportError:
        print(f"⚠ ONNX not available for validation")
    except Exception as e:
        print(f"⚠ ONNX validation failed: {e}")


def create_metadata_dict(cfg: PolicyConfig) -> dict:
    """Create metadata dictionary for openpilot compatibility.
    
    Returns metadata in the same format as the reference files.
    """
    # Calculate output dimensions
    plan_dim = cfg.plan_output_dim
    curv_dim = cfg.desired_curv_output_dim
    desire_dim = cfg.desire_pred_width
    
    # Total output dimension
    total_output_dim = plan_dim + curv_dim + desire_dim
    
    metadata = {
        "model_checkpoint": "custom-policy/trained",
        "output_slices": {
            "plan": slice(0, plan_dim),
            "desired_curvature": slice(plan_dim, plan_dim + curv_dim),
            "desire_state": slice(plan_dim + curv_dim, total_output_dim),
        },
        "input_shapes": {
            "features_buffer": (1, cfg.history_len, cfg.feature_len),
            "desire": (1, cfg.history_len, cfg.desire_len),
            "traffic_convention": (1, cfg.traffic_convention_len),
            "lateral_control_params": (1, cfg.lateral_control_params_len),
            "prev_desired_curv": (1, cfg.history_len, cfg.prev_desired_curv_len),
        },
        "output_shapes": {
            "outputs": (1, total_output_dim),
        },
    }
    
    return metadata


def save_metadata(cfg: PolicyConfig, output_path: str) -> None:
    """Save metadata pickle file for openpilot compatibility."""
    import pickle
    
    metadata = create_metadata_dict(cfg)
    
    with open(output_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Metadata saved to {output_path}")
    print(f"  Input shapes: {metadata['input_shapes']}")
    print(f"  Output shapes: {metadata['output_shapes']}")
    print(f"  Output slices: {list(metadata['output_slices'].keys())}")
