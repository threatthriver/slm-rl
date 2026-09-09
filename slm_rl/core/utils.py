"""Mathematical and tensor utilities for RL with language models."""

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selective_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Extract per-token log-probabilities for target input_ids using fused negative cross-entropy.
    Significantly faster and avoids allocating full (B, T, V) probability tensors in VRAM.
    """
    b, s, v = logits.shape
    flat_loss = F.cross_entropy(
        logits.reshape(-1, v),
        input_ids.reshape(-1),
        reduction="none",
    )
    return -flat_loss.reshape(b, s)



def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    Compute mean of tensor values where mask is 1.
    """
    mask_float = mask.to(tensor.dtype)
    if dim is None:
        return (tensor * mask_float).sum() / (mask_float.sum() + 1e-8)
    return (tensor * mask_float).sum(dim=dim) / (mask_float.sum(dim=dim) + 1e-8)


def masked_sum(tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    Compute sum of tensor values where mask is 1.
    """
    mask_float = mask.to(tensor.dtype)
    if dim is None:
        return (tensor * mask_float).sum()
    return (tensor * mask_float).sum(dim=dim)


def compute_kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    method: str = "low_var_kl",
    clamp_max: float = 10.0,
) -> torch.Tensor:
    """
    Compute token-level KL divergence between active policy and reference policy.
    
    Methods:
        - "low_var_kl": Schulman's non-negative, low-variance estimator:
          ratio = exp(ref_log_probs - log_probs)
          kl = ratio - 1 - log(ratio)
        - "kl": standard log ratio:
          kl = log_probs - ref_log_probs
        - "abs": absolute difference: |log_probs - ref_log_probs|
        - "mse": 0.5 * (log_probs - ref_log_probs)^2
    """
    log_ratio = ref_log_probs - log_probs  # log(pi_ref / pi)
    
    if method == "low_var_kl":
        # Guard against float32 exp overflow by clamping log_ratio to [-10, 10]
        clamped_log_ratio = torch.clamp(log_ratio, -clamp_max, clamp_max)
        ratio = torch.exp(clamped_log_ratio)
        kl = ratio - 1.0 - clamped_log_ratio
    elif method == "kl":
        kl = log_probs - ref_log_probs
    elif method == "abs":
        kl = torch.abs(log_probs - ref_log_probs)
    elif method == "mse":
        kl = 0.5 * (log_probs - ref_log_probs) ** 2
    else:
        raise ValueError(f"Unknown KL penalty method: {method}")
        
    return kl



def standardize(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Standardize tensor to zero mean and unit variance.
    If mask is provided, only consider unmasked elements for stats.
    """
    if mask is not None:
        mean = masked_mean(tensor, mask)
        var = masked_mean((tensor - mean) ** 2, mask)
        std = torch.sqrt(var + eps)
        return (tensor - mean) / std
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + eps)
