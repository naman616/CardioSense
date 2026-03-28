"""
Module: app/utils/inference.py

Responsibility:
    Model loading and inference pipeline for the Streamlit application.

Functions:
    load_model()     — Load trained ResNet1D from checkpoint file.
    run_inference()  — Preprocess a raw ECG signal and run model prediction.

Inference Pipeline (single sample):
    Raw numpy (187,)
    → Z-score normalization (same as training preprocessing)
    → Reshape to (1, 187, 1)
    → Convert to float32 tensor
    → Move to device
    → model.forward() → logits (1, 5)
    → Softmax → probabilities (1, 5)
    → argmax → predicted class index

Design Notes:
    - Z-score normalization uses PER-SAMPLE statistics (mean/std of the 187 values)
      matching the training preprocessing — no stored training statistics needed.
    - Model loaded via torch.load with weights_only=True for security.
    - Returns structured predictions dict for use by UI components.
    - Inference runs on CPU in the Streamlit app (no GPU assumption).
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"]
DEFAULT_CHECKPOINT = "results/checkpoints/resnet1d_adam_best.pth"


def load_model(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: torch.device | None = None,
) -> nn.Module:
    """Load trained ResNet1D from checkpoint.

    Returns:
        model: ResNet1D in eval mode on the specified device.
    """
    raise NotImplementedError


def run_inference(model: nn.Module, ecg_signal: np.ndarray) -> dict:
    """Preprocess and classify a single ECG beat.

    Args:
        model: Trained ResNet1D in eval mode.
        ecg_signal: Raw ECG values, shape (187,).

    Returns:
        dict with keys:
            'class_idx'    : int (0-4)
            'class_name'   : str
            'confidence'   : float (max softmax probability)
            'probabilities': np.ndarray shape (5,)
    """
    raise NotImplementedError
