"""Base reward class for RL training."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch


class BaseRewardFunction(ABC):
    """Abstract interface for reward calculation."""

    @abstractmethod
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        targets: Optional[List[Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of prompts and completions.
        
        Args:
            prompts: List of prompt strings
            completions: List of generated completion strings
            targets: Optional list of ground truth targets / answers
            
        Returns:
            rewards: 1D Tensor of scalar rewards, shape (batch_size,)
        """
        pass
