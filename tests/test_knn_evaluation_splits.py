"""Tests for K-fold split helpers in evaluate_knn."""

import math

import torch

from pilotwimae.downstream.beam_prediction.evaluate_knn import (
    _balanced_fold_segments,
    _mean_std,
)


def test_mean_std_unbiased_sample_std():
    out = _mean_std([1.0, 2.0, 3.0])
    assert out["mean"] == 2.0
    assert math.isclose(out["std"], 1.0)


def test_mean_std_single_value_std_zero():
    out = _mean_std([5.0])
    assert out["mean"] == 5.0
    assert out["std"] == 0.0


def test_balanced_folds_partition_indices():
    gen = torch.Generator().manual_seed(123)
    n, k = 103, 10
    segs = _balanced_fold_segments(n, k, gen)
    assert len(segs) == k
    sizes = [len(s) for s in segs]
    assert sum(sizes) == n
    assert max(sizes) - min(sizes) <= 1
    merged = torch.cat(segs).sort().values
    assert merged.tolist() == list(range(n))
