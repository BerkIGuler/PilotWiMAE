"""Masked NMSE on non-pilot mask (one scalar per channel tensor)."""

from __future__ import annotations

import numpy as np

from baselines.channel_prediction.masks import non_pilot_mask_from_pilots
from baselines.channel_prediction.metrics import nmse, nmse_masked


def test_nmse_masked_excludes_pilots() -> None:
    T, Na, Nf = 4, 2, 8
    rng = np.random.default_rng(0)
    target = rng.standard_normal((T, Na, Nf)) + 1j * rng.standard_normal((T, Na, Nf))
    pred = target.copy()
    pilot_times = [1, 3]
    known_sc = [0, 2, 4]
    m = non_pilot_mask_from_pilots(
        pilot_times=pilot_times,
        known_subcarriers=known_sc,
        T=T,
        N_a=Na,
        N_f=Nf,
    )
    assert nmse_masked(pred, target, m) < 1e-12

    pred2 = pred.copy()
    pred2[~m] += 10.0 + 10.0j
    assert abs(nmse_masked(pred2, target, m)) < 1e-12

    pred3 = pred.copy()
    pred3[m] += 1.0
    assert nmse_masked(pred3, target, m) > nmse(pred3, target)


def test_nmse_masked_matches_full_when_all_true() -> None:
    rng = np.random.default_rng(1)
    pred = rng.standard_normal((3, 4, 5)) + 1j * rng.standard_normal((3, 4, 5))
    tgt = rng.standard_normal((3, 4, 5)) + 1j * rng.standard_normal((3, 4, 5))
    m = np.ones(pred.shape, dtype=bool)
    assert abs(nmse_masked(pred, tgt, m) - nmse(pred, tgt)) < 1e-12
