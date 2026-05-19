"""
Masking module for PilotWiMAE.

Classes
-------
MaskGenerator
    Supports three strategies:
    - "random":   uniform random masking of individual patches.
    - "temporal": mask entire time steps (all patches in a selected time index).
    - "tube":     mask or keep full time-tubes at fixed (spatial, frequency)
                  locations across all time steps.

FactorizedMaskGenerator
    Tube + shared-frame mask for factorized attention:
    selects Tk time indices (tube) and Sk shared spatial positions per frame.
"""

from typing import Tuple

import torch


def factorized_flat_keep_from_t_s(
    ids_t_keep: torch.Tensor,
    ids_s_keep: torch.Tensor,
    ns_nf: int,
) -> torch.Tensor:
    """
    Flat patch indices for the factorized MAE mask (same broadcast as
    :class:`FactorizedMaskGenerator`).

    Args:
        ids_t_keep: ``(B, Tk)`` int64 — time indices per batch.
        ids_s_keep: ``(B, Sk)`` int64 — per-frame spatial indices in ``[0, ns_nf)``.
        ns_nf: ``ns * nf`` (patches per time step).

    Returns:
        ``(B, Tk * Sk)`` int64 — time-major flat indices into the ``(B, P, D)`` token
        sequence with ``P = nt * ns_nf``.
    """
    if ids_t_keep.dim() != 2 or ids_s_keep.dim() != 2:
        raise ValueError("ids_t_keep and ids_s_keep must be 2D (B, Tk) and (B, Sk)")
    if ids_t_keep.shape[0] != ids_s_keep.shape[0]:
        raise ValueError("Batch dimension mismatch between ids_t_keep and ids_s_keep")
    return (
        ids_t_keep.unsqueeze(-1) * int(ns_nf) + ids_s_keep.unsqueeze(1)
    ).reshape(ids_t_keep.shape[0], -1)


class MaskGenerator:
    """Configurable mask generator for 3D patch sequences."""

    def __init__(
        self,
        device: torch.device,
        mask_ratio: float = 0.6,
        strategy: str = "random",
        grid_dims: Tuple[int, int, int] = (1, 1, 1),
        random_seed: int = 42,
    ):
        """
        Args:
            device:     Torch device.
            mask_ratio: Fraction of patches (or structured units) to mask.
            strategy:   "random", "temporal", or "tube".
            grid_dims:  (nt, ns, nf) — number of patches per axis.
                        Required for "temporal" and "tube" strategies.
            random_seed: Seed for reproducibility.
        """
        self.mask_ratio = mask_ratio
        self.strategy = strategy
        self.grid_dims = grid_dims
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(
            random_seed if random_seed is not None else torch.seed()
        )

    def __call__(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, P, D) — embedded (or raw) patch sequence.

        Returns:
            unmasked: (B, P_keep, D) — kept patches.
            ids_keep: (B, P_keep)     — indices of kept patches.
            ids_mask: (B, P_mask)     — indices of masked patches.
        Notes:
            P_keep + P_mask = P
        """
        if self.strategy == "random":
            return self._random_mask(x)
        elif self.strategy == "temporal":
            return self._temporal_mask(x)
        elif self.strategy == "tube":
            return self._tube_mask(x)
        else:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")

    def _random_mask(self, x: torch.Tensor):
        B, P, D = x.shape
        num_keep = int(P * (1 - self.mask_ratio))

        noise = torch.rand(B, P, device=x.device, generator=self.generator)
        ids_shuffle = torch.argsort(noise, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        ids_mask = ids_shuffle[:, num_keep:]

        unmasked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        return unmasked, ids_keep, ids_mask

    def _temporal_mask(self, x: torch.Tensor):
        """Mask entire time steps.

        Patches are ordered time-major: patch index = it * (ns*nf) + is_ * nf + if_.
        Masking a time step means masking all ns*nf patches that share the same it.
        """
        B, P, D = x.shape
        nt, ns, nf = self.grid_dims
        patches_per_t = ns * nf

        if nt * patches_per_t != P:
            raise ValueError(
                f"grid_dims {self.grid_dims} imply {nt * patches_per_t} patches "
                f"but got P={P}"
            )

        num_t_keep = max(1, int(nt * (1 - self.mask_ratio)))

        noise = torch.rand(B, nt, device=x.device, generator=self.generator)
        ids_t_shuffle = torch.argsort(noise, dim=1)
        ids_t_keep = ids_t_shuffle[:, :num_t_keep]          # (B, num_t_keep)
        ids_t_mask = ids_t_shuffle[:, num_t_keep:]           # (B, num_t_mask)

        # Expand time indices to patch indices
        offsets = torch.arange(patches_per_t, device=x.device)  # (patches_per_t,)

        ids_keep = (
            ids_t_keep.unsqueeze(-1) * patches_per_t + offsets
        ).reshape(B, -1)  # (B, num_t_keep * patches_per_t)

        ids_mask = (
            ids_t_mask.unsqueeze(-1) * patches_per_t + offsets
        ).reshape(B, -1)  # (B, num_t_mask * patches_per_t)

        unmasked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        return unmasked, ids_keep, ids_mask

    def _tube_mask(self, x: torch.Tensor):
        """Mask entire time tubes at fixed (spatial, frequency) locations.

        Patch ordering is time-major:
            patch index = it * (ns*nf) + is_ * nf + if_
        A "tube" is all nt patches that share the same spatial-frequency index.
        """
        B, P, D = x.shape
        nt, ns, nf = self.grid_dims
        patches_per_t = ns * nf

        if nt * patches_per_t != P:
            raise ValueError(
                f"grid_dims {self.grid_dims} imply {nt * patches_per_t} patches "
                f"but got P={P}"
            )

        num_tubes = patches_per_t  # one tube per (spatial, frequency) location
        num_tubes_keep = max(1, int(num_tubes * (1 - self.mask_ratio)))

        # Sample which tubes (spatial-frequency positions) to keep.
        noise = torch.rand(B, num_tubes, device=x.device, generator=self.generator)
        ids_sf_shuffle = torch.argsort(noise, dim=1)
        ids_sf_keep = ids_sf_shuffle[:, :num_tubes_keep]   # (B, num_tubes_keep)
        ids_sf_mask = ids_sf_shuffle[:, num_tubes_keep:]   # (B, num_tubes_mask)

        # Expand tube indices over all time steps.
        time_offsets = torch.arange(nt, device=x.device) * patches_per_t  # (nt,)

        ids_keep = (
            ids_sf_keep.unsqueeze(-1) + time_offsets.unsqueeze(1)
        ).reshape(B, -1)  # (B, num_tubes_keep * nt)

        ids_mask = (
            ids_sf_mask.unsqueeze(-1) + time_offsets.unsqueeze(1)
        ).reshape(B, -1)  # (B, num_tubes_mask * nt)

        unmasked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        return unmasked, ids_keep, ids_mask


class FactorizedMaskGenerator:
    """
    Generates a tube + shared-frame mask for factorized attention.

    Patch ordering is assumed to be time-major:
        flat_index = it * (ns * nf) + spatial_index
    where ``spatial_index`` ranges over ``ns * nf`` patches per time step.

    Args:
        grid_dims:          ``(nt, ns, nf)`` — number of patches per axis.
        num_time_keep:      Number of time indices to keep (``Tk``).
        spatial_mask_ratio: Fraction of spatial patches to *mask* per frame.
        device:             Torch device.
        random_seed:        Seed for reproducibility.
    """

    def __init__(
        self,
        grid_dims: Tuple[int, int, int],
        num_time_keep: int,
        spatial_mask_ratio: float,
        device: torch.device,
        random_seed: int = 42,
    ):
        self.nt, self.ns, self.nf = grid_dims
        self.ns_nf = self.ns * self.nf
        self.num_time_keep = num_time_keep
        self.num_spatial_keep = max(1, round(self.ns_nf * (1.0 - spatial_mask_ratio)))
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(random_seed)

    def __call__(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(B, P, D)`` — embedded + pos-encoded tokens, where
               ``P = nt * ns_nf``.

        Returns:
            visible:  ``(B, Tk*Sk, D)`` — kept tokens.
            ids_keep: ``(B, Tk*Sk)``    — flat indices of kept patches.
            ids_mask: ``(B, P - Tk*Sk)``— flat indices of masked patches.
        """
        B, P, D = x.shape
        Tk = self.num_time_keep
        Sk = self.num_spatial_keep

        # --- sample temporal indices (same per batch element) ---
        t_noise = torch.rand(B, self.nt, device=self.device, generator=self.generator)
        ids_t_shuffle = torch.argsort(t_noise, dim=1)
        ids_t_keep = ids_t_shuffle[:, :Tk]   # (B, Tk)
        ids_t_mask = ids_t_shuffle[:, Tk:]   # (B, nt - Tk)

        # --- sample spatial indices (shared across all kept time steps) ---
        s_noise = torch.rand(B, self.ns_nf, device=self.device, generator=self.generator)
        ids_s_shuffle = torch.argsort(s_noise, dim=1)
        ids_s_keep = ids_s_shuffle[:, :Sk]   # (B, Sk)

        flat_keep = factorized_flat_keep_from_t_s(ids_t_keep, ids_s_keep, self.ns_nf)

        # --- compute flat ids_mask ---
        # All patches not in ids_keep.  We build the full set and subtract.
        all_ids = torch.arange(P, device=self.device).unsqueeze(0).expand(B, -1)  # (B, P)

        # Build a boolean mask of kept positions
        kept_mask = torch.zeros(B, P, dtype=torch.bool, device=self.device)
        kept_mask.scatter_(1, flat_keep, True)
        flat_mask = all_ids[~kept_mask].reshape(B, P - Tk * Sk)

        # --- gather visible tokens ---
        visible = torch.gather(
            x, dim=1, index=flat_keep.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, Tk*Sk, D)

        return visible, flat_keep, flat_mask
