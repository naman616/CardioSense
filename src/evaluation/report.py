"""
Module: src/evaluation/report.py

Responsibility:
    Generate a comprehensive evaluation report combining all metrics,
    plots, and optimizer comparison results into a structured summary.

Report Contents:
    1. Dataset summary (test set class distribution)
    2. Per-class metrics table (precision, recall, F1, AUC per class)
    3. Overall metrics (accuracy, macro F1, weighted F1, mean AUC)
    4. Confusion matrix (embedded image)
    5. ROC curves (embedded image)
    6. Optimizer comparison table (all 5 optimizers, final metrics)
    7. Adam paper connection: each metric grounded in a specific paper section

Design Notes:
    - Saves to results/reports/evaluation_report.txt (plain text for notebooks).
    - Also returns a pandas DataFrame of the per-class metrics table.
    - The optimizer comparison section explicitly cites Adam paper sections:
        SGD stalls → §6.1, AdaGrad stalls → §5, RMSProp diverges → §6.4,
        Adam wins → §6 (Theorem 4.1: O(√T) regret bound).
"""

import pandas as pd
from .metrics import MetricsResult

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]


def generate_report(
    metrics: MetricsResult,
    optimizer_histories: dict | None = None,
    save_path: str = "results/reports/evaluation_report.txt",
) -> pd.DataFrame:
    """Generate and save evaluation report.

    Returns:
        per_class_df: DataFrame with per-class metrics.
    """
    raise NotImplementedError
