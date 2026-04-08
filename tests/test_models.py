"""Tests for BaselineCNN and ResNet1D model architectures."""

import torch
import pytest
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet1d import ResNet1D


BATCH_SIZE = 8
INPUT_SHAPE = (BATCH_SIZE, 187, 1)  # (batch, length, channels)


class TestBaselineCNN:
    def test_output_shape(self):
        model = BaselineCNN(num_classes=5)
        x = torch.randn(*INPUT_SHAPE)
        # Conv1D expects (batch, channels, length) in PyTorch
        x = x.permute(0, 2, 1)
        out = model(x)
        assert out.shape == (BATCH_SIZE, 5)

    def test_output_is_logits(self):
        """Models output raw logits — values should not be constrained to [0,1]."""
        model = BaselineCNN(num_classes=5)
        model.eval()
        x = torch.randn(*INPUT_SHAPE).permute(0, 2, 1)
        with torch.no_grad():
            out = model(x)
        # Logits can be negative or > 1; softmax(logits) sums to 1
        sums = torch.softmax(out, dim=1).sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(BATCH_SIZE), atol=1e-5, rtol=0)
        # At least some logits should be outside [0, 1]
        assert out.min().item() < 0 or out.max().item() > 1, "Output looks like probabilities, not logits"


class TestResNet1D:
    def test_output_shape(self):
        model = ResNet1D(num_classes=5)
        x = torch.randn(*INPUT_SHAPE).permute(0, 2, 1)
        out = model(x)
        assert out.shape == (BATCH_SIZE, 5)

    def test_gradcam_target_layer_exists(self):
        model = ResNet1D(num_classes=5)
        layer = model.get_gradcam_target_layer()
        assert layer is not None

    def test_approx_param_count(self):
        model = ResNet1D(num_classes=5)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be approximately 790K (allow 50% tolerance)
        assert 400_000 < n_params < 1_200_000, f"Unexpected param count: {n_params}"
