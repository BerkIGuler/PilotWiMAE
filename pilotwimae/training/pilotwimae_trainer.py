"""
PilotWiMAE trainer implementation.
"""

import logging
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from .awgn import (
    awgn_complex_channel,
    snr_min_db_for_local_epoch,
    uniform_snr_db_per_sample,
)
from .trainer import BaseTrainer

logger = logging.getLogger(__name__)


class PilotWiMAETrainer(BaseTrainer):
    """Trainer for PilotWiMAE (masked reconstruction)."""

    def _noise_robust_snr_bounds(self) -> Optional[Tuple[float, float]]:
        """If noise_robust is enabled, return (snr_min_db, snr_max_db) for this epoch."""
        noise_cfg = (self.config.get("training") or {}).get("noise_robust") or {}
        if not bool(noise_cfg.get("enabled", False)):
            return None
        local_epoch = int(self.current_epoch - self._tensorboard_epoch_offset)
        total_epochs = int(self.config["training"]["epochs"])
        snr_start = float(noise_cfg.get("snr_start_db", 40.0))
        snr_max = float(noise_cfg.get("snr_max_db", 40.0))
        snr_min_db = snr_min_db_for_local_epoch(
            local_epoch, total_epochs, snr_start
        )
        return (snr_min_db, snr_max)

    def _patch_stats_targets(
        self, patches: torch.Tensor, *, eps: float
    ) -> torch.Tensor:
        """
        Return per-patch targets (mu, log_var) for patches shaped (B, P, D).
        """
        mu = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, unbiased=False, keepdim=True)
        log_var = (var + float(eps)).log()
        return torch.cat([mu, log_var], dim=-1)  # (B, P, 2)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_scale_enc = 0.0
        total_scale_dec = 0.0
        num_batches = len(train_loader)

        noise_bounds = self._noise_robust_snr_bounds()
        snr_min_db = None
        snr_max = 40.0
        if noise_bounds is not None:
            snr_min_db, snr_max = noise_bounds
            local_epoch = int(self.current_epoch - self._tensorboard_epoch_offset)
            logger.info(
                "noise_robust: global_epoch=%d local_epoch=%d snr_min_db=%.4f "
                "snr_max_db=%.4f (per-sample SNR ~ Uniform[snr_min, snr_max])",
                self.current_epoch,
                local_epoch,
                snr_min_db,
                snr_max,
            )

        progress_bar = tqdm(
            train_loader, desc=f"Training Epoch {self.current_epoch}"
        )

        for batch_idx, data in enumerate(progress_bar):
            data = data.to(self.device)

            self.optimizer.zero_grad()

            if noise_bounds is not None:
                B = data.shape[0]
                snr_db_batch = uniform_snr_db_per_sample(
                    B, snr_min_db, snr_max, self.device
                )
                data_encoder = awgn_complex_channel(data, snr_db_batch)
            else:
                data_encoder = data

            # Forward + loss (optionally under autocast for mixed precision).
            # AWGN is applied outside autocast (stable complex noise in float32).
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                # Targets from clean channel (identical to standard pretraining)
                patches = self.model.patcher(data)

                # Encoder/decoder see noisy channel when noise_robust is enabled
                output = self.model(
                    data_encoder, mask_ratio=self.model.mask_ratio
                )
                reconstructed_patches = output["reconstructed_patches"]
                ids_mask = output["ids_mask"]
                ids_keep = output["ids_keep"]

                # Compute reconstruction loss only on masked patches
                batch_size = patches.shape[0]

                loss_recon = torch.tensor(0.0, device=self.device, requires_grad=True)
                if ids_mask.shape[1] > 0:  # if there are masked patches
                    batch_indices = (
                        torch.arange(batch_size, device=self.device)
                        .unsqueeze(-1)
                        .expand(-1, ids_mask.shape[1])
                    )
                    recon_masked = reconstructed_patches[batch_indices, ids_mask]
                    target_masked = patches[batch_indices, ids_mask]

                    # MAE-style per-patch normalized loss (norm_pix_loss).
                    # Only the TARGET is normalized (zero mean, unit scale per patch).
                    # That weakens a trivial constant / mean-style reconstruction baseline
                    # in raw patch space, which would otherwise shrink MSE too easily.
                    if self.config["training"].get("norm_patch_loss", False):
                        eps = float(self.config["training"].get("norm_patch_loss_eps", 1e-6))
                        mean = target_masked.mean(dim=-1, keepdim=True)
                        var = target_masked.var(dim=-1, unbiased=False, keepdim=True)
                        target_masked = (target_masked - mean) / (var + eps).sqrt()

                    loss_recon = self.criterion(recon_masked, target_masked)

                loss = loss_recon

                # Optional auxiliary scale loss (mean + log-variance).
                use_scale = bool(getattr(self.model, "use_scale_loss", False))
                if use_scale:
                    sl_cfg = self.model.config.get("scale_loss", {}) if isinstance(self.model.config, dict) else {}
                    lam_enc = float(getattr(self.model, "scale_loss_lambda_enc", sl_cfg.get("lambda_enc", 0.1)))
                    lam_dec = float(getattr(self.model, "scale_loss_lambda_dec", sl_cfg.get("lambda_dec", 0.1)))
                    eps_s = float(getattr(self.model, "scale_loss_eps", sl_cfg.get("eps", 1e-8)))

                    # Targets for all patches (B, P, 2) in raw patch space.
                    tgt_all = self._patch_stats_targets(patches, eps=eps_s)

                    # --- encoder-side: visible only (ids_keep) ---
                    pred_enc = output.get("pred_scale_encoder", None)
                    if pred_enc is None:
                        raise RuntimeError("use_scale_loss=True but model did not return pred_scale_encoder.")
                    if pred_enc.shape[-1] != 2:
                        raise RuntimeError(f"pred_scale_encoder must have last dim 2, got {tuple(pred_enc.shape)}")
                    B, P_keep = ids_keep.shape
                    batch_idx_keep = torch.arange(B, device=self.device).unsqueeze(-1).expand(-1, P_keep)
                    tgt_enc = tgt_all[batch_idx_keep, ids_keep]
                    loss_scale_enc = torch.mean((pred_enc - tgt_enc) ** 2)

                    # --- decoder-side: masked only (ids_mask) ---
                    pred_dec_all = output.get("pred_scale_decoder", None)
                    if pred_dec_all is None:
                        raise RuntimeError("use_scale_loss=True but model did not return pred_scale_decoder.")
                    B2, P_mask = ids_mask.shape
                    batch_idx_mask = torch.arange(B2, device=self.device).unsqueeze(-1).expand(-1, P_mask)
                    tgt_dec = tgt_all[batch_idx_mask, ids_mask]
                    pred_dec = pred_dec_all[batch_idx_mask, ids_mask]
                    loss_scale_dec = torch.mean((pred_dec - tgt_dec) ** 2)

                    loss = loss + lam_enc * loss_scale_enc + lam_dec * loss_scale_dec
                else:
                    loss_scale_enc = torch.tensor(0.0, device=self.device)
                    loss_scale_dec = torch.tensor(0.0, device=self.device)

            # Backward with optional gradient scaling
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            gradient_clip_val = self.config["training"].get(
                "gradient_clip_val", 1.0
            )
            if gradient_clip_val > 0:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)

            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            total_loss += loss.item()
            total_recon += float(loss_recon.detach().item())
            total_scale_enc += float(loss_scale_enc.detach().item())
            total_scale_dec += float(loss_scale_dec.detach().item())
            self.global_step += 1

            postfix = {
                "loss": f"{loss.item():.4f}",
                "recon": f"{float(loss_recon.detach().item()):.4f}",
                "scale_enc": f"{float(loss_scale_enc.detach().item()):.4f}",
                "scale_dec": f"{float(loss_scale_dec.detach().item()):.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
            }
            if snr_min_db is not None:
                postfix["snr_min"] = f"{snr_min_db:.1f}"
            progress_bar.set_postfix(postfix)

            log_interval = self.config["logging"].get("log_every_n_steps", 100)
            if self.writer and batch_idx % log_interval == 0:
                self.writer.add_scalar(
                    "train/batch_loss", loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "train/loss_recon", float(loss_recon.detach().item()), self.global_step
                )
                self.writer.add_scalar(
                    "train/loss_scale_enc", float(loss_scale_enc.detach().item()), self.global_step
                )
                self.writer.add_scalar(
                    "train/loss_scale_dec", float(loss_scale_dec.detach().item()), self.global_step
                )

        metrics = {
            "train_loss": total_loss / num_batches,
            "train_loss_recon": total_recon / num_batches,
            "train_loss_scale_enc": total_scale_enc / num_batches,
            "train_loss_scale_dec": total_scale_dec / num_batches,
        }
        if snr_min_db is not None:
            metrics["snr_min_db"] = float(snr_min_db)
        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_recon = 0.0
        total_scale_enc = 0.0
        total_scale_dec = 0.0
        num_batches = len(val_loader)

        noise_bounds = self._noise_robust_snr_bounds()

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                data = data.to(self.device)

                if noise_bounds is not None:
                    snr_min_v, snr_max_v = noise_bounds
                    B = data.shape[0]
                    snr_db_batch = uniform_snr_db_per_sample(
                        B, snr_min_v, snr_max_v, self.device
                    )
                    data_encoder = awgn_complex_channel(data, snr_db_batch)
                else:
                    data_encoder = data

                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    patches = self.model.patcher(data)

                    output = self.model(
                        data_encoder, mask_ratio=self.model.mask_ratio
                    )
                    reconstructed = output["reconstructed_patches"]
                    ids_mask = output["ids_mask"]
                    ids_keep = output["ids_keep"]

                    loss_recon = torch.tensor(0.0, device=self.device)
                    if ids_mask.shape[1] > 0:
                        batch_size = patches.shape[0]
                        batch_indices = (
                            torch.arange(batch_size, device=self.device)
                            .unsqueeze(-1)
                            .expand(-1, ids_mask.shape[1])
                        )
                        recon_masked = reconstructed[batch_indices, ids_mask]
                        target_masked = patches[batch_indices, ids_mask]

                        if self.config["training"].get("norm_patch_loss", False):
                            eps = float(self.config["training"].get("norm_patch_loss_eps", 1e-6))
                            mean = target_masked.mean(dim=-1, keepdim=True)
                            var = target_masked.var(dim=-1, unbiased=False, keepdim=True)
                            target_masked = (target_masked - mean) / (var + eps).sqrt()

                        loss_recon = self.criterion(recon_masked, target_masked)

                    loss = loss_recon

                    use_scale = bool(getattr(self.model, "use_scale_loss", False))
                    if use_scale:
                        sl_cfg = self.model.config.get("scale_loss", {}) if isinstance(self.model.config, dict) else {}
                        lam_enc = float(getattr(self.model, "scale_loss_lambda_enc", sl_cfg.get("lambda_enc", 0.1)))
                        lam_dec = float(getattr(self.model, "scale_loss_lambda_dec", sl_cfg.get("lambda_dec", 0.1)))
                        eps_s = float(getattr(self.model, "scale_loss_eps", sl_cfg.get("eps", 1e-8)))

                        tgt_all = self._patch_stats_targets(patches, eps=eps_s)

                        pred_enc = output.get("pred_scale_encoder", None)
                        pred_dec_all = output.get("pred_scale_decoder", None)
                        if pred_enc is None or pred_dec_all is None:
                            raise RuntimeError("use_scale_loss=True but model did not return scale predictions.")

                        B, P_keep = ids_keep.shape
                        batch_idx_keep = torch.arange(B, device=self.device).unsqueeze(-1).expand(-1, P_keep)
                        tgt_enc = tgt_all[batch_idx_keep, ids_keep]
                        loss_scale_enc = torch.mean((pred_enc - tgt_enc) ** 2)

                        B2, P_mask = ids_mask.shape
                        batch_idx_mask = torch.arange(B2, device=self.device).unsqueeze(-1).expand(-1, P_mask)
                        tgt_dec = tgt_all[batch_idx_mask, ids_mask]
                        pred_dec = pred_dec_all[batch_idx_mask, ids_mask]
                        loss_scale_dec = torch.mean((pred_dec - tgt_dec) ** 2)

                        loss = loss + lam_enc * loss_scale_enc + lam_dec * loss_scale_dec
                    else:
                        loss_scale_enc = torch.tensor(0.0, device=self.device)
                        loss_scale_dec = torch.tensor(0.0, device=self.device)

                total_loss += loss.item()
                total_recon += float(loss_recon.detach().item())
                total_scale_enc += float(loss_scale_enc.detach().item())
                total_scale_dec += float(loss_scale_dec.detach().item())

        return {
            "val_loss": total_loss / num_batches,
            "val_loss_recon": total_recon / num_batches,
            "val_loss_scale_enc": total_scale_enc / num_batches,
            "val_loss_scale_dec": total_scale_dec / num_batches,
        }
