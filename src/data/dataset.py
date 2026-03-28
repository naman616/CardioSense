"""
Module: src/data/dataset.py

Responsibility:
    PyTorch Dataset and DataLoader wrappers for the preprocessed ECG data.

Classes:
    ECGDataset  — wraps numpy arrays into torch.utils.data.Dataset.
                  Converts (N, 187, 1) numpy arrays to float32 tensors
                  and integer labels to long tensors.

Design Notes:
    - Input arrays arrive already preprocessed (normalized, SMOTE-balanced,
      reshaped to (N, 187, 1)) from preprocessor.py.
    - DataLoaders use pin_memory=True when CUDA is available for faster
      host-to-device transfer during training.
    - Shuffle=True for train loader; False for val/test loaders.
    - Recommended batch size: 128 (per §8.6).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):
    """PyTorch Dataset for preprocessed ECG heartbeat arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def build_dataloaders(
    splits: dict,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, test DataLoaders from preprocessed splits dict.

    Returns:
        train_loader, val_loader, test_loader
    """
    raise NotImplementedError
