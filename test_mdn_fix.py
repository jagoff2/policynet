#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openpilot'))

import torch
from mdn import mdn_log_prob

def test_mdn_fix():
    print("=== Testing MDN Log Probability Fix ===")
    
    # Test with the exact parameters that were causing the error
    cfg_plan_mhp_n = 5
    batch_size = 1
    action_dim = 2
    
    # Create test parameters 
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
    
    try:
        # This should work now with the fix
        log_prob = mdn_log_prob(action, flat_params, cfg_plan_mhp_n)
        print(f"\n✓ SUCCESS! mdn_log_prob computed without error")
        print(f"  log_prob shape: {log_prob.shape}")
        print(f"  log_prob value: {log_prob}")
        
        # Test with different batch sizes
        print(f"\n=== Testing with batch_size=3 ===")
        batch_size_test = 3
        flat_params_batch = flat_params.repeat(batch_size_test, 1)
        action_batch = torch.randn(batch_size_test, action_dim)
        
        log_prob_batch = mdn_log_prob(action_batch, flat_params_batch, cfg_plan_mhp_n)
        print(f"✓ Batch test successful")
        print(f"  log_prob_batch shape: {log_prob_batch.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_mdn_fix()
    if success:
        print(f"\n🎉 MDN fix validated! Training should work now.")
    else:
        print(f"\n❌ MDN fix failed. More work needed.")
    exit(0 if success else 1)