"""
Module: src/explainability/gradcam.py

Responsibility:
    Gradient-weighted Class Activation Mapping (Grad-CAM) on the 1D ECG
    ResNet using PyTorch Captum (§8.9).

Method:
    1. Register forward hook on the final convolutional layer
       (ResNet1D's 128-filter ResidualBlock group).
    2. Run forward pass for a single ECG sample → get predicted class score.
    3. Backpropagate gradients to the target layer.
    4. Average gradients over the temporal dimension → channel weights α_k.
    5. Weighted sum of activation maps → 1D CAM heatmap.
    6. ReLU → keep only positive contributions.
    7. Normalize to [0, 1] for visualization.

Clinical Verification:
    - Class 1 (Supraventricular): heatmap should peak at P-wave region
      (early in the 187-sample window, before QRS onset).
    - Class 2 (Ventricular): heatmap should peak at QRS complex region
      (wide, aberrant QRS morphology is the defining feature).
    - This confirms the model attends to clinically relevant signal regions.

Design Notes:
    - Uses Captum's GradCam for 1D signal compatibility.
    - Target layer: model.get_gradcam_target_layer() (final conv block).
    - Returns raw 1D heatmap array (shape: 187 after upsampling) for overlay.
    - Requires model.eval() and input with grad enabled (no torch.no_grad()).
"""

import torch
import torch.nn as nn
import numpy as np


class GradCAM1D:
    """Grad-CAM implementation for 1D convolutional ECG models via Captum."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Trained ResNet1D in eval mode.
            target_layer: The convolutional layer to compute CAM from.
                          Should be model.get_gradcam_target_layer().
        """
        raise NotImplementedError

    def compute(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single ECG sample.

        Args:
            x: Input tensor of shape (1, 187, 1).
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            heatmap: 1D array of shape (187,), values in [0, 1].
        """
        raise NotImplementedError

    def _upsample_cam(self, cam: np.ndarray, target_length: int = 187) -> np.ndarray:
        """Upsample CAM from feature map size back to input signal length."""
        raise NotImplementedError
