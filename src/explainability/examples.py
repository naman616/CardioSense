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
from pathlib import Path
from .gradcam import GradCAM1D
from .heatmap_overlay import overlay_heatmap

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]
CLASS_ABBR = ["N", "S", "V", "F", "Q"]


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
    gradcam = GradCAM1D(model, model.get_gradcam_target_layer())
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    summary = {}

    # X_test shape: (N, 187, 1) — need to convert to (N, 1, 187) tensors
    for cls in range(5):
        cls_summary = {}

        # ── Correct example: highest confidence correct prediction ──────
        correct_mask = (y_true == cls) & (y_pred == cls)
        if correct_mask.any():
            correct_indices = np.where(correct_mask)[0]
            confidences = y_prob[correct_indices, cls]
            best_idx = correct_indices[np.argmax(confidences)]

            signal = X_test[best_idx, :, 0]   # (187,)
            x_tensor = torch.tensor(
                X_test[best_idx].T,  # (1, 187)
                dtype=torch.float32,
            ).unsqueeze(0).to(device)           # (1, 1, 187)

            heatmap = gradcam.compute(x_tensor, target_class=cls)
            correct_path = str(Path(save_dir) / f"gradcam_{CLASS_ABBR[cls]}_correct.png")
            fig = overlay_heatmap(signal, heatmap, cls, int(y_true[best_idx]),
                                  float(y_prob[best_idx, cls]), save_path=correct_path)
            fig.clf()
            cls_summary["correct"] = correct_path
        else:
            cls_summary["correct"] = None

        # ── Misclassified example: lowest-confidence wrong prediction ───
        wrong_mask = (y_true == cls) & (y_pred != cls)
        if wrong_mask.any():
            wrong_indices = np.where(wrong_mask)[0]
            # Confidence in wrong predicted class (model most confused = lowest correct prob)
            correct_probs = y_prob[wrong_indices, cls]
            worst_idx = wrong_indices[np.argmin(correct_probs)]

            signal = X_test[worst_idx, :, 0]  # (187,)
            x_tensor = torch.tensor(
                X_test[worst_idx].T,
                dtype=torch.float32,
            ).unsqueeze(0).to(device)

            pred_cls = int(y_pred[worst_idx])
            heatmap = gradcam.compute(x_tensor, target_class=pred_cls)
            wrong_path = str(Path(save_dir) / f"gradcam_{CLASS_ABBR[cls]}_misclassified.png")
            fig = overlay_heatmap(signal, heatmap, pred_cls, int(y_true[worst_idx]),
                                  float(y_prob[worst_idx, pred_cls]), save_path=wrong_path)
            fig.clf()
            cls_summary["misclassified"] = wrong_path
        else:
            cls_summary["misclassified"] = None

        summary[CLASS_NAMES[cls]] = cls_summary
        print(f"  {CLASS_NAMES[cls]}: correct={cls_summary['correct']}, "
              f"misclassified={cls_summary['misclassified']}")

    return summary
