"""Tests for kNN pilot_visible pilot string parsing and flat_keep geometry."""

import pytest
import torch

from pilotwimae.downstream.beam_prediction.pilot_pattern import (
    parse_pilot_pattern,
    pilot_visible_flat_keep,
)
from pilotwimae.models.modules.masking import factorized_flat_keep_from_t_s


def test_factorized_flat_keep_matches_manual_broadcast() -> None:
    ids_t = torch.tensor([[2, 11], [0, 1]], dtype=torch.long)
    ids_s = torch.tensor([[5, 3, 7], [2, 4, 6]], dtype=torch.long)
    ns_nf = 64
    manual = (ids_t.unsqueeze(-1) * ns_nf + ids_s.unsqueeze(1)).reshape(ids_t.shape[0], -1)
    out = factorized_flat_keep_from_t_s(ids_t, ids_s, ns_nf)
    assert out.shape == manual.shape
    assert torch.equal(out, manual)


def test_parse_pilot_pattern_basic() -> None:
    t, f = parse_pilot_pattern("t:2,11;f:1,6")
    assert t == [2, 11]
    assert f == [1, 6]


def test_parse_pilot_pattern_whitespace() -> None:
    t, f = parse_pilot_pattern(" t: 2 , 11 ; f: 1 , 6 ")
    assert t == [2, 11]
    assert f == [1, 6]


def test_parse_pilot_pattern_errors() -> None:
    with pytest.raises(ValueError, match="exactly one ';'"):
        parse_pilot_pattern("t:1,2")
    with pytest.raises(ValueError, match="t:"):
        parse_pilot_pattern("x:1;f:2")
    with pytest.raises(ValueError, match="f:"):
        parse_pilot_pattern("t:1;x:2")
    with pytest.raises(ValueError, match="at least one"):
        parse_pilot_pattern("t:;f:1")
    with pytest.raises(ValueError, match="Invalid integer"):
        parse_pilot_pattern("t:1,a;f:2")


def test_pilot_visible_flat_keep_example_14_8_8() -> None:
    """t:2,11; f:1,6 on (nt,ns,nf)=(14,8,8) => Tk=2, Sk=16, 32 flat indices."""
    dev = torch.device("cpu")
    flat = pilot_visible_flat_keep(14, 8, 8, [2, 11], [1, 6], device=dev)
    assert flat.shape == (1, 32)
    # Unique global indices
    u = torch.unique(flat)
    assert u.numel() == 32
    # In range [0, 896)
    assert int(flat.min()) >= 0
    assert int(flat.max()) < 14 * 64


def test_pilot_visible_flat_keep_time_index_oob() -> None:
    with pytest.raises(ValueError, match="Time index"):
        pilot_visible_flat_keep(4, 2, 2, [9], [0], device=torch.device("cpu"))


def test_pilot_visible_flat_keep_freq_oob() -> None:
    with pytest.raises(ValueError, match="Frequency index"):
        pilot_visible_flat_keep(4, 2, 2, [0], [3], device=torch.device("cpu"))


def test_pilot_visible_matches_factorized_mask_formula() -> None:
    nt, ns, nf = 5, 3, 4
    t_list = [1, 3]
    f_list = [0, 2]
    dev = torch.device("cpu")
    flat = pilot_visible_flat_keep(nt, ns, nf, t_list, f_list, device=dev)
    t_uniq = sorted(set(t_list))
    spatial = []
    for if_ in sorted(set(f_list)):
        for is_ in range(ns):
            spatial.append(is_ * nf + if_)
    ids_t = torch.tensor([t_uniq], device=dev, dtype=torch.long)
    ids_s = torch.tensor([spatial], device=dev, dtype=torch.long)
    ref = factorized_flat_keep_from_t_s(ids_t, ids_s, ns * nf)
    assert torch.equal(flat, ref)
