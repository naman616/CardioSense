"""
Module: src/data/splitter.py

Responsibility:
    Stratified train/validation split from the mitbih_train.csv data.

Design Notes:
    - Validation fraction: 15% (per §8.2 Step 5).
    - Uses stratified split to maintain per-class proportions in val set.
    - SMOTE is applied AFTER splitting to prevent leakage into val set.
    - Test set (mitbih_test.csv) is kept fully separate and untouched
      until final model evaluation in Step 8.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split into train and validation.

    Returns:
        X_train, X_val, y_train, y_val
    """
    return train_test_split(
        X, y,
        test_size=val_fraction,
        stratify=y,
        random_state=random_state,
    )
