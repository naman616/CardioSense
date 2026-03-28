"""
Module: src/eda/statistics.py

Responsibility:
    Compute per-class signal statistics to motivate preprocessing decisions.

Computed Statistics:
    - Mean amplitude (per class)
    - Signal variance (per class)         — motivates Z-score normalization
    - Beat length distribution            — motivates IQR outlier removal
    - Amplitude range (min, max, IQR)     — reveals inter-patient variability

Design Notes:
    - High variance in amplitude across patients (different ECG devices,
      electrode placement) directly motivates per-sample Z-score normalization.
    - Outlier beats (very short/long signal length) suggest digitization
      artifacts from Holter recordings, motivating IQR-based removal.
    - Output is a pandas DataFrame for display in notebooks and reports.
"""

import numpy as np
import pandas as pd

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unknown (Q)",
]


def compute_class_statistics(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Compute per-class signal statistics.

    Returns:
        DataFrame with columns: class, count, mean_amplitude, variance,
        min, max, iqr, beat_length_mean, beat_length_std
    """
    raise NotImplementedError
