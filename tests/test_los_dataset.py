"""Tests for pilotwimae.downstream.los datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pilotwimae.data import OptimizedPreloadedDataset
from pilotwimae.downstream.los.datasets import LosBinaryLabelDataset, load_los_binary_labels


def _save_npz_los(
    path: Path,
    *,
    s: int,
    n: int,
    t: int,
    k: int,
    los: np.ndarray,
    extra_meta: dict | None = None,
) -> None:
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((s, 1, 1, n, t, k)) + 1j * rng.standard_normal((s, 1, 1, n, t, k))).astype(
        np.complex64
    )
    shape = np.array([s, 1, 1, n, t, k], dtype=np.int64)
    meta: dict = {"los_binary": np.asarray(los, dtype=bool)}
    if extra_meta:
        meta.update(extra_meta)
    np.savez(path, h=h, shape=shape, metadata=np.array(meta, dtype=object))


def test_load_los_labels_aligns_with_preloaded(tmp_path: Path) -> None:
    s, n, t, k = 7, 2, 3, 4
    los = np.array([True, False, True, False, True, False, True])
    p = tmp_path / "one.npz"
    _save_npz_los(p, s=s, n=n, t=t, k=k, los=los)

    npz_files = [str(p)]
    labels = load_los_binary_labels(npz_files)
    ds = OptimizedPreloadedDataset(npz_files=npz_files, statistics=None)

    assert labels.shape == (s,)
    assert len(ds) == s
    assert torch.equal(labels, torch.tensor([1, 0, 1, 0, 1, 0, 1], dtype=torch.int64))

    wrapped = LosBinaryLabelDataset(ds, labels)
    ch, y = wrapped[2]
    assert ch.shape == (t, n, k)
    assert y.item() == 1


def test_load_los_concatenates_multiple_files(tmp_path: Path) -> None:
    p1 = tmp_path / "a.npz"
    p2 = tmp_path / "b.npz"
    _save_npz_los(p1, s=2, n=2, t=2, k=2, los=np.array([False, True]))
    _save_npz_los(p2, s=3, n=2, t=2, k=2, los=np.array([True, True, False]))

    npz_files = sorted([str(p1), str(p2)])
    labels = load_los_binary_labels(npz_files)
    ds = OptimizedPreloadedDataset(npz_files=npz_files, statistics=None)
    assert len(labels) == 5
    assert len(ds) == 5
    expected = torch.tensor([0, 1, 1, 1, 0], dtype=torch.int64)
    assert torch.equal(labels, expected)


def test_dataloader_batch_shapes(tmp_path: Path) -> None:
    s, n, t, k = 4, 2, 2, 2
    los = np.array([0, 1, 0, 1], dtype=bool)
    p = tmp_path / "d.npz"
    _save_npz_los(p, s=s, n=n, t=t, k=k, los=los)
    npz_files = [str(p)]
    labels = load_los_binary_labels(npz_files)
    ds = LosBinaryLabelDataset(OptimizedPreloadedDataset(npz_files, statistics=None), labels)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    xb, yb = next(iter(loader))
    assert xb.shape == (2, t, n, k)
    assert yb.shape == (2,)
    assert yb.dtype == torch.int64


def test_missing_los_binary_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.npz"
    rng = np.random.default_rng(0)
    s, n, t, k = 2, 2, 2, 2
    h = (rng.standard_normal((s, 1, 1, n, t, k)) + 1j * rng.standard_normal((s, 1, 1, n, t, k))).astype(
        np.complex64
    )
    shape = np.array([s, 1, 1, n, t, k], dtype=np.int64)
    np.savez(p, h=h, shape=shape, metadata=np.array({}, dtype=object))
    with pytest.raises(ValueError, match="los_binary"):
        load_los_binary_labels([str(p)])


def test_los_length_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad2.npz"
    s, n, t, k = 3, 2, 2, 2
    los = np.array([True, False], dtype=bool)
    _save_npz_los(p, s=s, n=n, t=t, k=k, los=los)
    with pytest.raises(ValueError, match="file_samples"):
        load_los_binary_labels([str(p)])


def test_wrapped_len_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.npz"
    _save_npz_los(p, s=2, n=2, t=2, k=2, los=np.array([True, False]))
    ds = OptimizedPreloadedDataset([str(p)], statistics=None)
    bad_labels = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(ValueError, match="len\\(base\\)"):
        LosBinaryLabelDataset(ds, bad_labels)
