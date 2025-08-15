#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openpilot'))

import torch
from mdn import mdn_split_params

def debug_mdn_shapes():
    print("=== Debugging MDN Tensor Shapes ===")
    
    # Simulate the exact parameters from the training
    cfg_plan_mhp_n = 5  # plan_mhp_n
    batch_size = 1
    
    # Create mock tensors matching what the training code produces
    # From the training: combined_mu/log_std are (B, n_mix, 2)
    combined_mu = torch.randn(batch_size, cfg_plan_mhp_n, 2)
    combined_log_std = torch.randn(batch_size, cfg_plan_mhp_n, 2) * 0.1
    logits_comb = torch.randn(batch_size, cfg_plan_mhp_n)
    
    print(f"Input tensor shapes:")
    print(f"  combined_mu: {combined_mu.shape}")
    print(f"  combined_log_std: {combined_log_std.shape}")
    print(f"  logits_comb: {logits_comb.shape}")
    
    # Apply the fix from train.py
    flat_mu = combined_mu.view(combined_mu.shape[0], -1)
    flat_log_std = combined_log_std.view(combined_log_std.shape[0], -1)
    flat_params = torch.cat([flat_mu, flat_log_std, logits_comb], dim=-1)
    
    print(f"\nAfter flattening:")
    print(f"  flat_mu: {flat_mu.shape}")
    print(f"  flat_log_std: {flat_log_std.shape}")
    print(f"  flat_params: {flat_params.shape}")
    
    # Create action tensor (curvature, acceleration)
    action = torch.randn(batch_size, 2)
    print(f"  action: {action.shape}")
    
    # Now test mdn_split_params
    print(f"\n=== Testing mdn_split_params ===")
    try:
        mu, log_std, logits = mdn_split_params(flat_params, cfg_plan_mhp_n)
        print(f"mdn_split_params output:")
        print(f"  mu: {mu.shape}")
        print(f"  log_std: {log_std.shape}")
        print(f"  logits: {logits.shape}")
        
        # Test the problematic computation from mdn_log_prob
        print(f"\n=== Testing mdn_log_prob computation ===")
        x_exp = action.unsqueeze(-2)  # shape (..., 1, d)
        print(f"  x_exp: {x_exp.shape}")
        print(f"  mu: {mu.shape}")
        print(f"  log_std: {log_std.shape}")
        
        var = torch.exp(2.0 * log_std)
        print(f"  var: {var.shape}")
        
        # This is where the error occurs
        diff = x_exp - mu
        print(f"  diff (x_exp - mu): {diff.shape}")
        
        sq = (diff ** 2) / (2.0 * var)
        print(f"  sq: {sq.shape}")
        
        # The error happens here: sq.sum(-1)
        sq_sum = sq.sum(-1)
        print(f"  sq.sum(-1): {sq_sum.shape}")
        
        print("✓ No error - tensor shapes are compatible")
        
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Print detailed tensor info for debugging
        print(f"\nDetailed tensor information:")
        if 'mu' in locals():
            print(f"  mu.shape: {mu.shape}")
        if 'x_exp' in locals():
            print(f"  x_exp.shape: {x_exp.shape}")
        if 'diff' in locals():
            print(f"  diff.shape: {diff.shape}")

if __name__ == "__main__":
    debug_mdn_shapes()