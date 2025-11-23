import random
import torch
from typing import Tuple


def set_seed(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy for predictions (logits) and integer targets."""
    if preds.dim() == 2:
        preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0
