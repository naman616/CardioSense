"""
Module: src/optimizers/convergence_plots.py

Responsibility:
    Generate convergence visualization plots from the optimizer comparison
    study results, grounding each observation in Adam paper theory.

Plots Generated:
    1. Training Loss Curves (all 5 optimizers on one axes)
       - Linked to: SGD stalls (Adam §6.1), AdaGrad slow late (§5),
                    RMSProp divergence (§6.4), Adam fastest (§6).

    2. Validation Macro F1 Curves (all 5 optimizers)
       - Shows Adam's superior minority-class learning rate.

    3. Per-Class F1 Comparison Bar Chart (final epoch, all optimizers)
       - Highlights Adam's advantage on Fusion Beat (Class 3, 0.7%),
         which has the sparsest gradients.

    4. Convergence Speed Table
       - Epochs to reach 90% of final val F1 per optimizer.
       - Wall-clock time to convergence.

Design Notes:
    - All plots use consistent color scheme per optimizer across all figures.
    - Adam is highlighted in a distinct color (e.g., red/orange).
    - Each plot annotation references the specific Adam paper section.
    - Saved to results/plots/training_curves/{metric}_{optimizers}.png.
"""

import matplotlib.pyplot as plt
import numpy as np


OPTIMIZER_COLORS = {
    "sgd": "#1f77b4",           # blue
    "sgd_momentum": "#ff7f0e",  # orange
    "adagrad": "#2ca02c",       # green
    "rmsprop": "#9467bd",       # purple
    "adam": "#d62728",          # red — proposed method
}


def plot_training_loss_curves(histories: dict, save_path: str | None = None) -> None:
    """Plot training loss over epochs for all optimizers."""
    raise NotImplementedError


def plot_val_f1_curves(histories: dict, save_path: str | None = None) -> None:
    """Plot validation macro F1 over epochs for all optimizers."""
    raise NotImplementedError


def plot_per_class_f1_comparison(histories: dict, save_path: str | None = None) -> None:
    """Bar chart of final per-class F1 for all optimizers."""
    raise NotImplementedError


def plot_convergence_speed(histories: dict, threshold: float = 0.90) -> None:
    """Print/plot epochs and wall-clock time to reach threshold% of final F1."""
    raise NotImplementedError
