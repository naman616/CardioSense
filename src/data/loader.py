"""
Module: src/data/loader.py

Responsibility:
    Load the MIT-BIH and PTB-DB CSV datasets from disk into pandas DataFrames.
    Separates features (187 ECG signal values) from the class label column.

Inputs:
    - data/raw/mitbih_train.csv  (87,554 rows x 188 cols)
    - data/raw/mitbih_test.csv   (21,892 rows x 188 cols)
    - data/raw/ptbdb_normal.csv
    - data/raw/ptbdb_abnormal.csv

Outputs:
    - X: np.ndarray of shape (N, 187)  — raw signal values
    - y: np.ndarray of shape (N,)      — integer class labels 0-4

Design Notes:
    - Last column is the label; all preceding 187 columns are signal.
    - PTB-DB loader concatenates normal+abnormal and assigns binary labels (0/1).
    - No preprocessing happens here — this is pure I/O.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_mitbih(
    train_path: str = "data/raw/mitbih_train.csv",
    test_path: str = "data/raw/mitbih_test.csv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MIT-BIH train and test splits.

    Returns:
        X_train, y_train, X_test, y_test
    """
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)
    X_train = df_train.iloc[:, :187].values.astype(np.float32)
    y_train = df_train.iloc[:, 187].values.astype(int)
    X_test = df_test.iloc[:, :187].values.astype(np.float32)
    y_test = df_test.iloc[:, 187].values.astype(int)
    return X_train, y_train, X_test, y_test


def load_ptbdb(
    normal_path: str = "data/raw/ptbdb_normal.csv",
    abnormal_path: str = "data/raw/ptbdb_abnormal.csv",
) -> tuple[np.ndarray, np.ndarray]:
    """Load PTB-DB dataset (binary: 0=healthy, 1=myocardial infarction).

    Returns:
        X, y
    """
    df_normal = pd.read_csv(normal_path, header=None)
    df_abnormal = pd.read_csv(abnormal_path, header=None)
    df_normal = df_normal.iloc[:, :187]
    df_normal["label"] = 0
    df_abnormal = df_abnormal.iloc[:, :187]
    df_abnormal["label"] = 1
    df = pd.concat([df_normal, df_abnormal], ignore_index=True)
    X = df.iloc[:, :187].values.astype(np.float32)
    y = df.iloc[:, 187].values.astype(int)
    return X, y
