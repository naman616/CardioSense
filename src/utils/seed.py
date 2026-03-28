"""
Module: src/utils/seed.py

Responsibility:
    Set global random seeds for reproducibility across all runs.
    Critical for the optimizer comparison study (§8.7) — all models
    must start from identical weight initializations.

Seeds Set:
    - Python random
    - NumPy random
    - PyTorch CPU random
    - PyTorch CUDA random (all GPUs)
    - torch.backends.cudnn deterministic mode

Design Notes:
    - Default seed: 42 (used throughout the project).
    - CUDA determinism (cudnn.deterministic=True) trades some speed
      for exact reproducibility of GPU operations.
    - Called at the top of every training script and notebook.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
