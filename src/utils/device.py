"""
Module: src/utils/device.py

Responsibility:
    Detect and return the best available compute device.

Priority:
    1. CUDA (NVIDIA GPU) — RTX 4070 Laptop (~10 min training)
    2. MPS (Apple Silicon) — fallback for development
    3. CPU — final fallback

Design Notes:
    - Returns torch.device for use in model.to(device) and tensor.to(device).
    - Prints device info on first call for training confirmation.
"""

import torch


def get_device(verbose: bool = True) -> torch.device:
    """Return best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU")
    return device
