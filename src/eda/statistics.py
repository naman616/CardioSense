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
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        cls_X = X[y == i]
        if len(cls_X) == 0:
            rows.append({
                "class": name, "count": 0,
                "mean_amplitude": np.nan, "variance": np.nan,
                "min": np.nan, "max": np.nan, "iqr": np.nan,
                "beat_length_mean": np.nan, "beat_length_std": np.nan,
            })
            continue

        # Flatten all signal values for global stats per class
        flat = cls_X.ravel()
        q1, q3 = np.percentile(flat, [25, 75])

        # Beat "length" estimated as number of non-near-zero samples (proxy for beat width)
        beat_lengths = np.sum(np.abs(cls_X) > 0.01, axis=1).astype(float)

        rows.append({
            "class": name,
            "count": len(cls_X),
            "mean_amplitude": float(np.mean(flat)),
            "variance": float(np.var(flat)),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "iqr": float(q3 - q1),
            "beat_length_mean": float(beat_lengths.mean()),
            "beat_length_std": float(beat_lengths.std()),
        })

    df = pd.DataFrame(rows)
    df = df.set_index("class")
    return df
