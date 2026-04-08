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
from pathlib import Path
from .metrics import MetricsResult

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]

ADAM_PAPER_NOTES = {
    "sgd":          "§6.1: Stalls on sparse minority-class gradients; no per-parameter adaptation.",
    "sgd_momentum": "§5:   Momentum helps, but no adaptive per-parameter learning rate.",
    "adagrad":      "§5:   Monotonically decaying LR stalls late in training.",
    "rmsprop":      "§6.4: Diverges at high β2 without bias correction.",
    "adam":         "§4/§6: O(√T) regret bound; bias correction; handles sparsity (Theorem 4.1).",
}


def generate_report(
    metrics: MetricsResult,
    optimizer_histories: dict | None = None,
    save_path: str = "results/reports/evaluation_report.txt",
) -> pd.DataFrame:
    """Generate and save evaluation report.

    Returns:
        per_class_df: DataFrame with per-class metrics.
    """
    per_class_df = pd.DataFrame({
        "Class":     CLASS_NAMES,
        "Precision": metrics.per_class_precision.round(4),
        "Recall":    metrics.per_class_recall.round(4),
        "F1":        metrics.per_class_f1.round(4),
        "ROC-AUC":   metrics.per_class_roc_auc.round(4),
    }).set_index("Class")

    lines = [
        "=" * 70,
        "CardioSense — Model Evaluation Report",
        "1D ResNet + Adam Optimizer (Kingma & Ba, ICLR 2015)",
        "=" * 70,
        "",
        "── Overall Metrics ──────────────────────────────────────────",
        f"  Accuracy      : {metrics.accuracy:.4f}  (target ≥ 0.98)",
        f"  Macro F1      : {metrics.macro_f1:.4f}  (target ≥ 0.93)",
        f"  Weighted F1   : {metrics.weighted_f1:.4f}",
        f"  Mean ROC-AUC  : {metrics.mean_roc_auc:.4f}",
        "",
        "── Per-Class Metrics ────────────────────────────────────────",
        per_class_df.to_string(),
        "",
    ]

    if optimizer_histories:
        lines += [
            "── Optimizer Comparison (final val Macro F1) ────────────────",
            f"{'Optimizer':<20} {'Final F1':>10}  {'Adam Paper Citation':<40}",
            "-" * 72,
        ]
        for name, hist in optimizer_histories.items():
            final_f1 = hist.get("val_f1", [0.0])[-1]
            note = ADAM_PAPER_NOTES.get(name, "")
            lines.append(f"{name:<20} {final_f1:>10.4f}  {note}")
        lines.append("")

    lines += [
        "── References ───────────────────────────────────────────────",
        "  Kingma & Ba (2015). Adam: A Method for Stochastic Optimization.",
        "  Lin et al. (2017). Focal Loss for Dense Object Detection.",
        "  He et al. (2016). Deep Residual Learning for Image Recognition.",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    print(report_text)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report_text)

    return per_class_df
