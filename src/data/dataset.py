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
        # X shape: (N, 187, 1) → permute to (N, 1, 187) for PyTorch Conv1d
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, 1, 187)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_dataloaders(
    splits: dict,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, test DataLoaders from preprocessed splits dict.

    Returns:
        train_loader, val_loader, test_loader
    """
    pin = torch.cuda.is_available()
    train_ds = ECGDataset(splits["X_train"], splits["y_train"])
    val_ds   = ECGDataset(splits["X_val"],   splits["y_val"])
    test_ds  = ECGDataset(splits["X_test"],  splits["y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=0)

    return train_loader, val_loader, test_loader
