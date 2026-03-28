"""Tests for data loading, preprocessing, and dataset modules."""

import numpy as np
import pytest
import torch
from src.data.preprocessor import normalize, reshape_for_conv1d


class TestNormalize:
    def test_output_shape_unchanged(self):
        X = np.random.randn(100, 187).astype(np.float32)
        X_norm = normalize(X)
        assert X_norm.shape == X.shape

    def test_per_sample_mean_zero(self):
        X = np.random.randn(50, 187).astype(np.float32)
        X_norm = normalize(X)
        means = X_norm.mean(axis=1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)

    def test_per_sample_std_one(self):
        X = np.random.randn(50, 187).astype(np.float32)
        X_norm = normalize(X)
        stds = X_norm.std(axis=1)
        np.testing.assert_allclose(stds, 1.0, atol=1e-5)


class TestReshapeForConv1D:
    def test_output_shape(self):
        X = np.random.randn(100, 187).astype(np.float32)
        X_reshaped = reshape_for_conv1d(X)
        assert X_reshaped.shape == (100, 187, 1)

    def test_values_unchanged(self):
        X = np.random.randn(10, 187).astype(np.float32)
        X_reshaped = reshape_for_conv1d(X)
        np.testing.assert_array_equal(X_reshaped[:, :, 0], X)
