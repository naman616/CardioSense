"""Tests for evaluation metrics computation."""

import numpy as np
import pytest
from src.evaluation.metrics import compute_metrics


class TestComputeMetrics:
    def _make_perfect_predictions(self, n: int = 100, num_classes: int = 5):
        y_true = np.random.randint(0, num_classes, n)
        y_pred = y_true.copy()
        y_prob = np.zeros((n, num_classes))
        y_prob[np.arange(n), y_true] = 1.0
        return y_true, y_pred, y_prob

    def test_perfect_accuracy(self):
        y_true, y_pred, y_prob = self._make_perfect_predictions()
        result = compute_metrics(y_true, y_pred, y_prob)
        assert result.accuracy == pytest.approx(1.0)

    def test_perfect_macro_f1(self):
        y_true, y_pred, y_prob = self._make_perfect_predictions()
        result = compute_metrics(y_true, y_pred, y_prob)
        assert result.macro_f1 == pytest.approx(1.0)

    def test_per_class_f1_shape(self):
        y_true, y_pred, y_prob = self._make_perfect_predictions()
        result = compute_metrics(y_true, y_pred, y_prob)
        assert result.per_class_f1.shape == (5,)

    def test_roc_auc_shape(self):
        y_true, y_pred, y_prob = self._make_perfect_predictions()
        result = compute_metrics(y_true, y_pred, y_prob)
        assert result.per_class_roc_auc.shape == (5,)

    def test_auc_in_valid_range(self):
        y_true, y_pred, y_prob = self._make_perfect_predictions()
        result = compute_metrics(y_true, y_pred, y_prob)
        assert all(0.0 <= auc <= 1.0 for auc in result.per_class_roc_auc)
