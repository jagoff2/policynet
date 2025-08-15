#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openpilot'))

import torch
from mdn import mdn_split_params, mdn_log_prob

def debug_exact_error():
    print("=== Reproducing Exact Training Error ===")
    
    # Let's check what happens with the exact parameter calculation format
    # This simulates the problematic code from train.py extract_action_and_log_prob
    
    cfg_plan_mhp_n = 5
    batch_size = 1
    
    print("Testing with exact parameter format from mdn_split_params...")
    
    # Test different parameter formats to see which one causes the issue
    
    # Format 1: Our attempted fix - grouped format [all_mus, all_log_stds, all_logits]
    print("\n=== Format 1: Grouped [mus, log_stds, logits] ===")
    combined_mu = torch.randn(batch_size, cfg_plan_mhp_n, 2)
    combined_log_std = torch.randn(batch_size, cfg_plan_mhp_n, 2) * 0.1
    logits_comb = torch.randn(batch_size, cfg_plan_mhp_n)
    
    flat_mu = combined_mu.view(combined_mu.shape[0], -1)  # (1, 10)
    flat_log_std = combined_log_std.view(combined_log_std.shape[0], -1)  # (1, 10) 
    flat_params_v1 = torch.cat([flat_mu, flat_log_std, logits_comb], dim=-1)  # (1, 25)
    
    print(f"flat_params shape: {flat_params_v1.shape}")
    
    try:
        mu1, log_std1, logits1 = mdn_split_params(flat_params_v1, cfg_plan_mhp_n)
        print(f"SUCCESS - mu: {mu1.shape}, log_std: {log_std1.shape}, logits: {logits1.shape}")
        
        # Test the full log_prob computation
        action = torch.randn(batch_size, 2)
        log_prob = mdn_log_prob(action, flat_params_v1, cfg_plan_mhp_n)
        print(f"log_prob computed successfully: {log_prob.shape}")
        
    except Exception as e:
        print(f"ERROR in Format 1: {e}")
        print(f"Error type: {type(e).__name__}")
    
    # Format 2: Check what mdn_split_params expects for multi-mode
    print("\n=== Format 2: Expected multi-mode format ===")
    # For multi-mode: d = (full_dim - n_mix) // (2 * n_mix)
    # So for n_mix=5: d = (25 - 5) // (2 * 5) = 20 // 10 = 2 ✓
    # Expected: params should be viewable as (batch, n_mix, 2*d + 1) = (1, 5, 5)
    
    d = 2  # action dimension
    expected_per_mix = 2 * d + 1  # 5 parameters per mixture
    total_expected = cfg_plan_mhp_n * expected_per_mix  # 25 total
    
    print(f"Expected: {cfg_plan_mhp_n} mixtures × {expected_per_mix} params = {total_expected} total")
    
    # Create properly formatted parameters for multi-mode
    # Format should be: [means, log_stds, logits] all flattened properly
    test_params = torch.randn(batch_size, total_expected)
    
    try:
        mu2, log_std2, logits2 = mdn_split_params(test_params, cfg_plan_mhp_n)
        print(f"SUCCESS - mu: {mu2.shape}, log_std: {log_std2.shape}, logits: {logits2.shape}")
        
        action = torch.randn(batch_size, 2)
        log_prob = mdn_log_prob(action, test_params, cfg_plan_mhp_n)
        print(f"log_prob computed successfully: {log_prob.shape}")
        
    except Exception as e:
        print(f"ERROR in Format 2: {e}")
        print(f"Error type: {type(e).__name__}")
    
    # Let's also inspect the mdn_split_params logic more carefully
    print(f"\n=== Analyzing mdn_split_params logic ===")
    print(f"For n_mix={cfg_plan_mhp_n}, full_dim=25:")
    full_dim = 25
    d_calc = (full_dim - cfg_plan_mhp_n) // (2 * cfg_plan_mhp_n)
    print(f"  Calculated d = ({full_dim} - {cfg_plan_mhp_n}) // (2 * {cfg_plan_mhp_n}) = {d_calc}")
    
    if d_calc != 2:
        print(f"  ⚠️  PROBLEM: Expected d=2 but calculated d={d_calc}")
        print(f"  This means our parameter format is wrong!")
    else:
        print(f"  ✓ d calculation is correct")

if __name__ == "__main__":
    debug_exact_error()