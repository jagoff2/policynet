#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openpilot'))

import torch
from mdn import mdn_split_params

def debug_mdn_log_prob_step_by_step():
    print("=== Debug mdn_log_prob Step by Step ===")
    
    # Create test data that reproduces the error
    cfg_plan_mhp_n = 5
    batch_size = 1
    action_dim = 2
    
    # Create parameters in the format that's causing issues
    combined_mu = torch.randn(batch_size, cfg_plan_mhp_n, action_dim)
    combined_log_std = torch.randn(batch_size, cfg_plan_mhp_n, action_dim) * 0.1
    logits_comb = torch.randn(batch_size, cfg_plan_mhp_n)
    
    flat_mu = combined_mu.view(batch_size, -1)  # (1, 10)
    flat_log_std = combined_log_std.view(batch_size, -1)  # (1, 10)
    flat_params = torch.cat([flat_mu, flat_log_std, logits_comb], dim=-1)  # (1, 25)
    
    action = torch.randn(batch_size, action_dim)  # (1, 2)
    
    print(f"Input shapes:")
    print(f"  flat_params: {flat_params.shape}")
    print(f"  action: {action.shape}")
    print(f"  n_mix: {cfg_plan_mhp_n}")
    
    # Step through mdn_log_prob manually to find the exact error
    try:
        # Step 1: mdn_split_params
        print(f"\n=== Step 1: mdn_split_params ===")
        mu, log_std, logits = mdn_split_params(flat_params, cfg_plan_mhp_n)
        print(f"  mu: {mu.shape}")
        print(f"  log_std: {log_std.shape}")
        print(f"  logits: {logits.shape}")
        
        # Step 2: Expand target to align with mixture components
        print(f"\n=== Step 2: Expand action ===")
        x_exp = action.unsqueeze(-2)  # shape (..., 1, d)
        print(f"  x_exp: {x_exp.shape}")
        
        # Step 3: Compute component log likelihoods
        print(f"\n=== Step 3: Compute variance ===")
        var = torch.exp(2.0 * log_std)
        print(f"  var: {var.shape}")
        
        # Step 4: Compute (x - mu)^2 / (2 * var)
        print(f"\n=== Step 4: Compute squared differences ===")
        diff = x_exp - mu
        print(f"  diff (x_exp - mu): {diff.shape}")
        
        sq_numerator = (diff ** 2)
        print(f"  sq_numerator: {sq_numerator.shape}")
        
        sq_denominator = 2.0 * var
        print(f"  sq_denominator: {sq_denominator.shape}")
        
        sq = sq_numerator / sq_denominator
        print(f"  sq: {sq.shape}")
        
        # Step 5: Sum over dimensions - THIS IS WHERE THE ERROR LIKELY OCCURS
        print(f"\n=== Step 5: Sum over dimensions ===")
        sq_sum = sq.sum(-1)
        print(f"  sq.sum(-1): {sq_sum.shape}")
        
        # Step 6: Compute log component probabilities
        print(f"\n=== Step 6: Compute log components ===")
        const_term = 0.5 * torch.log(2.0 * torch.tensor(torch.pi))
        print(f"  const_term: {const_term} (scalar)")
        
        log_comp = -log_std - const_term - sq_sum
        print(f"  ERROR: This line should fail with dimension mismatch")
        
    except Exception as e:
        print(f"\n✗ ERROR CAUGHT: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Let's check what's happening with the tensor operations
        print(f"\n=== Debugging the specific error ===")
        print(f"log_std shape: {log_std.shape}")
        print(f"sq_sum shape: {sq_sum.shape}")
        
        # The error suggests log_std and sq_sum have incompatible shapes
        # Let's see what happens if we try to combine them
        try:
            result1 = -log_std
            print(f"-log_std works, shape: {result1.shape}")
        except Exception as e1:
            print(f"Error in -log_std: {e1}")
            
        try:
            const_term = 0.5 * torch.log(2.0 * torch.tensor(torch.pi))
            result2 = -log_std - const_term
            print(f"-log_std - const_term works, shape: {result2.shape}")
        except Exception as e2:
            print(f"Error in -log_std - const_term: {e2}")
            
        try:
            result3 = -log_std - const_term - sq_sum
            print(f"Full expression works unexpectedly")
        except Exception as e3:
            print(f"Error in full expression: {e3}")
            print(f"This confirms the dimension mismatch issue")

if __name__ == "__main__":
    debug_mdn_log_prob_step_by_step()