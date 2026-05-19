import time

import pytest
import torch
from torch.utils.data import Dataset

from pilotwimae.data.beam.beam_codebook import (
    flatten_beam_index,
    generate_upa_2d_dft_codebook,
    unflatten_beam_index,
    upa_2d_dft_num_beams,
    upa_axis_dft_codewords,
)
from pilotwimae.data.beam.beam_labels import (
    beam_targets_from_gains,
    compute_beam_gains,
)
from pilotwimae.data.beam.datasets import (
    BeamLabelDatasetWrapper,
    collate_beam_targets,
)


class ToyChannelDataset(Dataset):
    def __init__(self, channels: torch.Tensor):
        self.channels = channels

    def __len__(self) -> int:
        return self.channels.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.channels[idx]


def test_codebook_shape_norms_and_index_mapping():
    n_h, n_v = 4, 2
    o_h, o_v = 2, 3
    codebook = generate_upa_2d_dft_codebook(n_h=n_h, n_v=n_v, o_h=o_h, o_v=o_v)
    m = (o_h * n_h) * (o_v * n_v)
    n = n_h * n_v

    assert codebook.shape == (m, n)
    norms = torch.linalg.norm(codebook, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    n_h_total = o_h * n_h
    m_h, m_v = 5, 3
    flat = flatten_beam_index(m_h=m_h, m_v=m_v, n_h_total=n_h_total)
    rec_h, rec_v = unflatten_beam_index(flat, n_h_total=n_h_total)
    assert rec_h == m_h
    assert rec_v == m_v


def test_undersampled_codebook_shape_norms_and_index_mapping():
    n_h, n_v = 8, 4
    u_h, u_v = 2, 2
    k_h = n_h // u_h
    k_v = n_v // u_v
    codebook = generate_upa_2d_dft_codebook(
        n_h=n_h, n_v=n_v, o_h=1, o_v=1, u_h=u_h, u_v=u_v
    )
    m = k_h * k_v
    n = n_h * n_v
    assert codebook.shape == (m, n)
    norms = torch.linalg.norm(codebook, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    m_h, m_v = 2, 1
    flat = flatten_beam_index(m_h=m_h, m_v=m_v, n_h_total=k_h)
    rec_h, rec_v = unflatten_beam_index(flat, n_h_total=k_h)
    assert rec_h == m_h
    assert rec_v == m_v


def test_mixed_axis_oversample_and_undersample():
    """Oversample along vertical; undersample along horizontal."""
    n_h, n_v = 8, 4
    codebook = generate_upa_2d_dft_codebook(
        n_h=n_h, n_v=n_v, o_h=1, o_v=2, u_h=2, u_v=1
    )
    k_h = n_h // 2
    k_v = n_v * 2
    assert codebook.shape == (k_h * k_v, n_h * n_v)


def test_upa_axis_dft_codewords_rejects_illegal_combine():
    with pytest.raises(ValueError, match="Cannot combine oversampling"):
        upa_axis_dft_codewords(8, o=2, u=2)


def test_upa_axis_dft_codewords_rejects_indivisible_u():
    with pytest.raises(ValueError, match="must divide"):
        upa_axis_dft_codewords(8, o=1, u=3)


def test_upa_2d_dft_num_beams_matches_generator():
    assert upa_2d_dft_num_beams(8, 4, u_h=2, u_v=2) == 8
    assert upa_2d_dft_num_beams(8, 4, o_h=1, o_v=1) == 32
    assert upa_2d_dft_num_beams(8, 4, o_h=2, o_v=2) == 128


def test_beam_targets_recover_known_codeword():
    n_h, n_v = 2, 2
    codebook = generate_upa_2d_dft_codebook(n_h=n_h, n_v=n_v, o_h=1, o_v=1)
    target_idx = 2

    t, k = 5, 7
    channel = codebook[target_idx][:, None].repeat(1, k)  # (N, K)
    channel = channel.unsqueeze(0).repeat(t, 1, 1)  # (T, N, K)

    gains = compute_beam_gains(channel, codebook)
    top1_indices, _ = beam_targets_from_gains(gains, label_mode="sequence", top_k=1)
    assert torch.all(top1_indices.squeeze(-1) == target_idx)

    snapshot_top1, _ = beam_targets_from_gains(gains, label_mode="snapshot", top_k=1)
    assert int(snapshot_top1.item()) == target_idx


def test_beam_targets_topk_shapes():
    torch.manual_seed(0)
    b, t, n, k = 3, 4, 8, 6
    m = 16
    channels = torch.randn(b, t, n, k, dtype=torch.complex64)
    codebook = torch.randn(m, n, dtype=torch.complex64)
    codebook = codebook / torch.linalg.norm(codebook, dim=1, keepdim=True)

    gains = compute_beam_gains(channels, codebook)
    seq_idx, seq_scores = beam_targets_from_gains(gains, label_mode="sequence", top_k=3)
    snap_idx, snap_scores = beam_targets_from_gains(gains, label_mode="snapshot", top_k=2)

    assert seq_idx.shape == (b, t, 3)
    assert seq_scores.shape == (b, t, 3)
    assert snap_idx.shape == (b, 2)
    assert snap_scores.shape == (b, 2)


def test_beam_label_dataset_wrapper_formats():
    n_h, n_v = 2, 2
    n = n_h * n_v
    s, t, k = 2, 3, 4
    codebook = generate_upa_2d_dft_codebook(n_h=n_h, n_v=n_v)
    target_idx = 1

    channels = torch.zeros(s, t, n, k, dtype=torch.complex64)
    channels[0] = codebook[target_idx][:, None].repeat(1, k).unsqueeze(0).repeat(t, 1, 1)
    channels[1] = codebook[0][:, None].repeat(1, k).unsqueeze(0).repeat(t, 1, 1)
    base_dataset = ToyChannelDataset(channels)

    ds_class = BeamLabelDatasetWrapper(
        base_dataset,
        n_h=n_h,
        n_v=n_v,
        label_mode="sequence",
        return_format="class_index",
        top_k=1,
    )
    batch_channels = [ds_class[0], ds_class[1]]
    x0, y0 = collate_beam_targets(batch_channels, dataset=ds_class)
    assert x0.shape == (s, t, n, k)
    assert y0.shape == (s, t)
    assert torch.all(y0[0] == target_idx)

    ds_topk = BeamLabelDatasetWrapper(
        base_dataset,
        n_h=n_h,
        n_v=n_v,
        label_mode="snapshot",
        return_format="topk_indices",
        top_k=2,
    )
    _, y1 = collate_beam_targets([ds_topk[0], ds_topk[1]], dataset=ds_topk)
    assert y1.shape == (s, 2)
    assert int(y1[0, 0]) == target_idx

    ds_dict = BeamLabelDatasetWrapper(
        base_dataset,
        n_h=n_h,
        n_v=n_v,
        label_mode="snapshot",
        return_format="indices_and_scores",
        top_k=2,
    )
    _, y2 = collate_beam_targets([ds_dict[0], ds_dict[1]], dataset=ds_dict)
    assert "indices" in y2 and "scores" in y2
    assert y2["indices"].shape == (s, 2)
    assert y2["scores"].shape == (s, 2)


def test_dataset_make_collate_fn():
    channels = torch.randn(2, 3, 4, 5, dtype=torch.complex64)
    ds = BeamLabelDatasetWrapper(
        ToyChannelDataset(channels),
        n_h=2,
        n_v=2,
        label_mode="snapshot",
        return_format="class_index",
        top_k=1,
    )
    collate_fn = ds.make_collate_fn()
    out_channels, out_targets = collate_fn([ds[0], ds[1]])

    assert out_channels.shape == (2, 3, 4, 5)
    assert out_targets.shape == (2,)


def test_collate_beam_targets_timing_bs512():
    """
    Measure batch collate+label generation time for batch size 512.

    This is a profiling-style test intended to inform downstream training strategy.
    """
    batch_size = 512
    t, n_h, n_v, k = 14, 8, 4, 32
    n = n_h * n_v
    channels = torch.randn(batch_size, t, n, k, dtype=torch.complex64)

    ds = BeamLabelDatasetWrapper(
        ToyChannelDataset(channels),
        n_h=n_h,
        n_v=n_v,
        o_h=1,
        o_v=1,
        label_mode="snapshot",
        return_format="class_index",
        top_k=1,
    )
    collate_fn = ds.make_collate_fn()

    batch = [ds[i] for i in range(batch_size)]
    warmup_channels, warmup_targets = collate_fn(batch)
    assert warmup_channels.shape == (batch_size, t, n, k)
    assert warmup_targets.shape == (batch_size,)

    n_runs = 5
    start = time.perf_counter()
    for _ in range(n_runs):
        out_channels, out_targets = collate_fn(batch)
    elapsed_s = time.perf_counter() - start

    assert out_channels.shape == (batch_size, t, n, k)
    assert out_targets.shape == (batch_size,)

    avg_ms = (elapsed_s / n_runs) * 1000.0
    assert avg_ms < 100.0, f"Average collate latency too high: {avg_ms:.2f} ms (expected < 100 ms)"
