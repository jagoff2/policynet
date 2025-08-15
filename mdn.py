"""
mdn.py
=======

This module implements utilities for working with mixture density networks
(MDNs) based on multivariate Gaussians. Openpilot represents many of its
model outputs (plan, curvature, lead, lanes) as the parameters of an MDN.
The helper functions here provide safe exponentiation, softmax operations
that avoid numerical overflow, and functions to compute log likelihoods
and negative log likelihood loss for supervised learning. Sampling is
included primarily for debugging; during policy training the means are
treated as deterministic actions.

The functions in this module operate on PyTorch tensors. They have been
written to be differentiable to support gradient-based optimisation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "safe_exp",
    "safe_softmax",
    "mdn_split_params",
    "mdn_log_prob",
    "mdn_nll_loss",
    "mdn_sample",
    "mdn_mean",
]


def safe_exp(x: Tensor) -> Tensor:
    """Exponentiate a tensor with clipping to avoid overflow.

    Openpilot clips exponentiation to roughly ``exp(11)`` when running on
    FP16 hardware. We replicate that behaviour here. See ``Parser.safe_exp``
    in openpilot for details.

    Args:
        x: Input tensor

    Returns:
        Tensor with exponentiation applied, with large values clipped.
    """
    # limit corresponds approximately to np.exp(11) ~ 59874
    return torch.exp(torch.clamp(x, max=11.0))


def safe_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax that can operate in FP16 without overflow.

    Args:
        logits: Input tensor of unnormalised log probabilities.
        dim: Dimension along which to apply softmax.

    Returns:
        Softmax-normalised probabilities summing to one along ``dim``.
    """
    # subtract max for numerical stability
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    exp = safe_exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def mdn_split_params(params: Tensor, n_mix: int) -> tuple[Tensor, Tensor, Tensor]:
    """Split MDN parameters matching openpilot's format.

    Openpilot MDN format:
    - Multi-mode: [mu1, mu2, ..., log_std1, log_std2, ..., logit1, logit2, ...]
    - Single-mode: [mu, log_std] (no logits)

    Args:
        params: Flattened MDN parameters
        n_mix: Number of mixture components (use 1 for single-mode)

    Returns:
        Tuple of (mu, log_std, logits)
    """
    *batch, full_dim = params.shape
    
    if n_mix == 1:
        # Single-mode MDN: just [mean, log_std]
        d = full_dim // 2
        params = params.view(*batch, 1, full_dim)
        mu = params[..., :d]
        log_std = params[..., d:2*d]
        logits = torch.zeros(*batch, 1, device=params.device)  # Dummy logits
    else:
        # Multi-mode MDN: [means, log_stds, logits] 
        d = (full_dim - n_mix) // (2 * n_mix)
        params = params.view(*batch, n_mix, 2 * d + 1)
        mu = params[..., :d]
        log_std = params[..., d:2*d]
        logits = params[..., -1]
    
    return mu, log_std, logits


def mdn_log_prob(x: Tensor, params: Tensor, n_mix: int) -> Tensor:
    """Compute the log probability of a target under a mixture of Gaussians.

    Args:
        x: Target tensor of shape ``(..., d)``. Should broadcast to match
           the batch dimensions of ``params``.
        params: Flattened MDN parameters of shape
            ``(..., n_mix * (2 * d + 1))``.
        n_mix: Number of mixture components.

    Returns:
        Log probability of shape ``(...)`` summing log probabilities over
        the mixture components.
    """
    mu, log_std, logits = mdn_split_params(params, n_mix)
    # Expand target to align with mixture components
    x_exp = x.unsqueeze(-2)  # shape (..., 1, d)
    # Compute component log likelihoods
    var = torch.exp(2.0 * log_std)
    # Compute (x - mu)^2 / (2 * var)
    sq = ((x_exp - mu) ** 2) / (2.0 * var)
    log_comp = -log_std.sum(-1) - 0.5 * torch.log(2.0 * torch.tensor(torch.pi)) - sq.sum(-1)
    # Weight by mixture weights
    log_weights = F.log_softmax(logits, dim=-1)
    log_probs = torch.logsumexp(log_weights + log_comp, dim=-1)
    return log_probs


def mdn_nll_loss(x: Tensor, params: Tensor, n_mix: int) -> Tensor:
    """Negative log likelihood loss for an MDN output.

    Args:
        x: Target tensor of shape ``(..., d)``.
        params: MDN parameter tensor of shape ``(..., n_mix * (2 * d + 1))``.
        n_mix: Number of mixture components.

    Returns:
        The mean negative log likelihood over the batch.
    """
    nll = -mdn_log_prob(x, params, n_mix)
    return nll.mean()


def mdn_sample(params: Tensor, n_mix: int) -> Tensor:
    """Sample from MDN matching openpilot behavior.
    
    For deployment, openpilot typically uses the mixture mean rather than sampling.
    This function provides sampling for training/exploration.
    """
    mu, log_std, logits = mdn_split_params(params, n_mix)
    
    if n_mix == 1:
        # Single mode: just use the mean + noise
        std = safe_exp(log_std)
        eps = torch.randn_like(std)
        return mu.squeeze(-2) + std.squeeze(-2) * eps
    else:
        # Multi-mode: sample mixture component then from that component
        weights = safe_softmax(logits, dim=-1)
        mix_idx = torch.distributions.Categorical(probs=weights).sample()
        
        # Gather selected parameters
        batch_indices = torch.arange(mu.shape[0], device=mu.device)
        selected_mu = mu[batch_indices, mix_idx]  # (B, d)
        selected_log_std = log_std[batch_indices, mix_idx]  # (B, d)
        
        std = safe_exp(selected_log_std)
        eps = torch.randn_like(std)
        return selected_mu + std * eps


def mdn_mean(params: Tensor, n_mix: int) -> Tensor:
    """Extract mixture mean for deterministic control.
    
    This matches openpilot's typical usage where the dominant mixture
    component mean is used for control commands.
    """
    mu, log_std, logits = mdn_split_params(params, n_mix)
    
    if n_mix == 1:
        return mu.squeeze(-2)
    else:
        # Use the highest-weight mixture component
        weights = safe_softmax(logits, dim=-1)
        max_idx = torch.argmax(weights, dim=-1)
        
        batch_indices = torch.arange(mu.shape[0], device=mu.device)
        return mu[batch_indices, max_idx]
