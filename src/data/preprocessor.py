"""
Module: src/data/preprocessor.py

Responsibility:
    Full 5-step preprocessing pipeline as specified in the proposal (§8.2).

Pipeline (in order):
    [1] Z-score normalization per sample
            x_norm = (x - mean(x)) / std(x)
            Removes patient-specific amplitude differences.
            Ensures uniform gradient magnitudes across batches (critical for Adam).

    [2] IQR-based outlier removal (training set only)
            Remove beats with signal lengths outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
            Eliminates Holter digitization artifacts.

    [3] SMOTE oversampling (training set ONLY — prevents data leakage)
            Uses imbalanced-learn's SMOTE to oversample minority classes.
            Balances all 5 classes prior to model training.
            Class distribution after: ~equal across N, S, V, F, Q.

    [4] Tensor reshape
            (N, 187)  -->  (N, 187, 1)  required for Conv1D input.

    [5] Dataset split
            Train       : mitbih_train.csv minus 15% held as validation
            Validation  : 15% of training set (early stopping / LR tuning)
            Test        : mitbih_test.csv (untouched until final evaluation)

Design Notes:
    - Steps [2] and [3] must only be applied to training data to prevent leakage.
    - Z-score stats (mean, std) computed on training set and applied to val/test.
    - SMOTE operates in raw feature space (not tensor space).
    - Returns numpy arrays; tensor conversion happens inside ECGDataset.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def normalize(X: np.ndarray) -> np.ndarray:
    """Per-sample Z-score normalization."""
    raise NotImplementedError


def remove_outliers(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IQR-based outlier removal on signal amplitude range."""
    raise NotImplementedError


def apply_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE oversampling to balance all 5 arrhythmia classes."""
    raise NotImplementedError


def reshape_for_conv1d(X: np.ndarray) -> np.ndarray:
    """Reshape (N, 187) -> (N, 187, 1) for PyTorch Conv1d input."""
    raise NotImplementedError


def preprocess(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    val_fraction: float = 0.15,
) -> dict:
    """Run the full pipeline and return train/val/test splits.

    Returns dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test
        (all as numpy arrays with final shape (N, 187, 1))
    """
    raise NotImplementedError
