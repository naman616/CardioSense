"""Tests for Focal Loss implementation."""

import torch
import numpy as np
import pytest
from src.training.focal_loss import FocalLoss, compute_class_frequencies


class TestFocalLoss:
    def test_output_is_scalar(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(32, 5)
        targets = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0  # scalar

    def test_loss_is_nonnegative(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(32, 5)
        targets = torch.randint(0, 5, (32,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_focal_downweights_easy_examples(self):
        """Well-classified examples should contribute less loss than hard examples."""
        loss_fn = FocalLoss(gamma=2.0)
        # Easy example: strong correct prediction
        easy_logits = torch.tensor([[10.0, -5.0, -5.0, -5.0, -5.0]])
        easy_target = torch.tensor([0])
        # Hard example: uniform/wrong prediction
        hard_logits = torch.tensor([[-5.0, 10.0, -5.0, -5.0, -5.0]])
        hard_target = torch.tensor([0])
        easy_loss = loss_fn(easy_logits, easy_target)
        hard_loss = loss_fn(hard_logits, hard_target)
        assert easy_loss.item() < hard_loss.item()

    def test_with_class_frequencies(self):
        freqs = np.array([0.83, 0.03, 0.06, 0.007, 0.073])
        loss_fn = FocalLoss(gamma=2.0, class_frequencies=freqs)
        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0


class TestComputeClassFrequencies:
    def test_sums_to_one(self):
        y = np.array([0] * 83 + [1] * 3 + [2] * 6 + [3] * 1 + [4] * 7)
        freqs = compute_class_frequencies(y, num_classes=5)
        assert abs(freqs.sum() - 1.0) < 1e-6

    def test_shape(self):
        y = np.random.randint(0, 5, 1000)
        freqs = compute_class_frequencies(y, num_classes=5)
        assert freqs.shape == (5,)
