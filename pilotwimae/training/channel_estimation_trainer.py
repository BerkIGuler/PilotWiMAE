"""
Trainer for supervised channel estimation with fixed pilot pattern.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pilotwimae.downstream.beam_prediction.pilot_pattern import (
    parse_pilot_pattern,
    pilot_visible_flat_keep,
)
from pilotwimae.models import PilotWiMAE
from pilotwimae.models.encoder_backbone import is_factorized_family

from .trainer import BaseTrainer

logger = logging.getLogger(__name__)


class ChannelEstimationTrainer(BaseTrainer):
    """MSE training on masked (non-pilot) patches using fixed pilot visibility."""

    def __init__(
        self,
        config: Dict,
        device: Optional[torch.device] = None,
    ):
        super().__init__(config, device=device)
        ce_cfg = self.config.get("task", {}).get("channel_estimation")
        if not isinstance(ce_cfg, dict):
            raise KeyError(
                "task.channel_estimation is required for ChannelEstimationTrainer."
            )
        pilot_pattern = ce_cfg.get("pilot_pattern")
        if not isinstance(pilot_pattern, str) or not pilot_pattern.strip():
            raise KeyError(
                "task.channel_estimation.pilot_pattern is required (e.g. 't:2,11;f:0,2,4,6')."
            )
        if not isinstance(self.model, PilotWiMAE):
            raise TypeError("ChannelEstimationTrainer requires a PilotWiMAE model instance.")

        nt, ns, nf = self.model.grid_dims
        t_nums, f_nums = parse_pilot_pattern(pilot_pattern)
        pilot_batch = pilot_visible_flat_keep(nt, ns, nf, t_nums, f_nums, device=self.device)
        self.pilot_flat_keep = pilot_batch.squeeze(0).to(self.device, dtype=torch.long)

        tk = len(sorted(set(t_nums)))
        sk = len(sorted(set(f_nums))) * int(ns)
        self.pilot_factorized_grid: Optional[Tuple[int, int]] = None
        if is_factorized_family(self.model.encoder_type):
            self.pilot_factorized_grid = (tk, sk)

        p_keep = int(self.pilot_flat_keep.numel())
        logger.info(
            "Loaded CE pilot pattern '%s' -> keep %d/%d patches",
            pilot_pattern,
            p_keep,
            int(self.model.num_patches),
        )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for data in tqdm(train_loader, desc=f"Train epoch {self.current_epoch}"):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                patches = self.model.patcher(data)
                output = self.model.reconstruct_pilot_masked(
                    data,
                    self.pilot_flat_keep,
                    pilot_factorized_grid=self.pilot_factorized_grid,
                )
                reconstructed = output["reconstructed_patches"]
                ids_mask = output["ids_mask"]

                if ids_mask.shape[1] == 0:
                    raise RuntimeError("No masked patches found for CE training loss.")
                batch_size = patches.shape[0]
                batch_indices = (
                    torch.arange(batch_size, device=self.device)
                    .unsqueeze(-1)
                    .expand(-1, ids_mask.shape[1])
                )
                recon_masked = reconstructed[batch_indices, ids_mask]
                target_masked = patches[batch_indices, ids_mask]
                loss = self.criterion(recon_masked, target_masked)

            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            gradient_clip_val = self.config["training"].get("gradient_clip_val", 1.0)
            if gradient_clip_val > 0:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clip_val
                )

            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

        return {"loss": total_loss / max(1, num_batches)}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                data = data.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    patches = self.model.patcher(data)
                    output = self.model.reconstruct_pilot_masked(
                        data,
                        self.pilot_flat_keep,
                        pilot_factorized_grid=self.pilot_factorized_grid,
                    )
                    reconstructed = output["reconstructed_patches"]
                    ids_mask = output["ids_mask"]

                    if ids_mask.shape[1] == 0:
                        raise RuntimeError("No masked patches found for CE validation loss.")
                    batch_size = patches.shape[0]
                    batch_indices = (
                        torch.arange(batch_size, device=self.device)
                        .unsqueeze(-1)
                        .expand(-1, ids_mask.shape[1])
                    )
                    recon_masked = reconstructed[batch_indices, ids_mask]
                    target_masked = patches[batch_indices, ids_mask]
                    loss = self.criterion(recon_masked, target_masked)

                total_loss += loss.item()

        return {"val_loss": total_loss / max(1, num_batches)}
