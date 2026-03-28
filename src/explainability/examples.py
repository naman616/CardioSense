"""
Module: src/explainability/examples.py

Responsibility:
    Select and generate Grad-CAM visualizations for representative
    correctly classified and misclassified examples per class (§8.9).

Selection Strategy:
    For each class (0-4):
        - Correctly classified: highest-confidence correct prediction.
        - Misclassified: lowest-confidence wrong prediction (model most
          confused — most informative for error analysis).

Output:
    - 10 Grad-CAM overlay figures (5 classes × 2 examples each).
    - Saved to results/plots/gradcam/.
    - Returns a summary dict for use in evaluation report.

Design Notes:
    - Requires both the test set predictions AND the trained model (for Grad-CAM).
    - Misclassified examples reveal which signal regions caused confusion,
      providing insight into model failure modes.
    - These examples are shown in the Streamlit reference gallery.
"""

import numpy as np
import torch
import torch.nn as nn
from .gradcam import GradCAM1D
from .heatmap_overlay import overlay_heatmap

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]


def get_class_examples(
    model: nn.Module,
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    device: torch.device,
    save_dir: str = "results/plots/gradcam",
) -> dict:
    """Generate and save Grad-CAM visualizations for all 5 classes.

    Returns:
        summary: dict with keys per class, each containing paths to
                 'correct' and 'misclassified' example figures.
    """
    raise NotImplementedError
