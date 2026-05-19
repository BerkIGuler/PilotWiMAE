"""
Beam prediction dataset wrappers.
"""

from functools import partial
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from .beam_codebook import generate_upa_2d_dft_codebook
from .beam_labels import beam_targets_from_gains, compute_beam_gains
from .types import LabelMode, ReturnFormat


class BeamLabelDatasetWrapper(Dataset):
    """
    Wrap a base channel dataset for batched on-the-fly beam label generation.

    Base dataset is required to return a complex channel tensor of shape
    (T, N, K) from __getitem__.

    This wrapper returns channels only from __getitem__. Beam targets are
    computed in batch via collate_fn to leverage vectorized beam scoring.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        n_h: int,
        n_v: int,
        o_h: int = 1,
        o_v: int = 1,
        u_h: int = 1,
        u_v: int = 1,
        antenna_order: Literal["hv", "vh"] = "hv",
        label_mode: LabelMode = "sequence",
        return_format: ReturnFormat = "class_index",
        top_k: int = 1,
        codebook_dtype: torch.dtype = torch.complex64,
    ):
        if n_h <= 0 or n_v <= 0:
            raise ValueError("n_h and n_v must be positive.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        self.base_dataset = base_dataset
        self.label_mode = label_mode
        self.return_format = return_format
        self.top_k = top_k

        self.codebook = generate_upa_2d_dft_codebook(
            n_h=n_h,
            n_v=n_v,
            o_h=o_h,
            o_v=o_v,
            u_h=u_h,
            u_v=u_v,
            antenna_order=antenna_order,
            dtype=codebook_dtype,
        )
        self.num_beams = self.codebook.size(0)
        if top_k > self.num_beams:
            raise ValueError(f"top_k={top_k} cannot be larger than number of beams={self.num_beams}.")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _format_target(self, indices: torch.Tensor, scores: torch.Tensor) -> Any:
        if self.return_format == "class_index":
            return indices[..., 0]
        if self.return_format == "topk_indices":
            return indices
        if self.return_format == "indices_and_scores":
            return {"indices": indices, "scores": scores}
        raise ValueError("return_format must be one of {'class_index', 'topk_indices', 'indices_and_scores'}.")

    def _extract_channel(self, sample: Any) -> torch.Tensor:
        if not isinstance(sample, torch.Tensor):
            raise ValueError("Base dataset sample must be a complex torch.Tensor with shape (T, N, K).")

        channel = sample

        if channel.ndim != 3:
            raise ValueError(f"Expected channel shape (T, N, K), got {tuple(channel.shape)}.")
        if not channel.dtype.is_complex:
            raise ValueError("Channel tensor must be complex.")
        if channel.size(1) != self.codebook.size(1):
            raise ValueError(
                f"Channel antenna dimension N={channel.size(1)} does not match codebook N={self.codebook.size(1)}."
            )
        return channel

    def compute_batch_targets(self, channels: torch.Tensor) -> Any:
        """
        Compute beam targets for a batch of channels.

        Args:
            channels: Complex tensor with shape (B, T, N, K).
        """
        codebook = self.codebook.to(device=channels.device, dtype=channels.dtype)
        gains = compute_beam_gains(channels, codebook)
        indices, scores = beam_targets_from_gains(gains, label_mode=self.label_mode, top_k=self.top_k)
        return self._format_target(indices=indices, scores=scores)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.base_dataset[idx]
        channel = self._extract_channel(sample)
        return channel

    def make_collate_fn(self):
        """Return a collate_fn bound to this dataset for DataLoader use."""
        return partial(collate_beam_targets, dataset=self)


def collate_beam_targets(
    batch: list[torch.Tensor],
    *,
    dataset: BeamLabelDatasetWrapper,
) -> tuple[torch.Tensor, Any]:
    """
    Collate channels and generate beam targets in batch for DataLoader.

    Intended to be passed as `collate_fn` to `torch.utils.data.DataLoader`,
    typically via `dataset.make_collate_fn()`.
    """
    channels = torch.stack(batch, dim=0)
    targets = dataset.compute_batch_targets(channels)
    return channels, targets
