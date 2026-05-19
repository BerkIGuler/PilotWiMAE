"""
kNN regression on embeddings: predict continuous (B, D) targets (e.g. scalar distance or R^D vectors).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class kNNforRegression:
    """
    kNN regressor: weighted average of neighbor targets in embedding space.

    Neighbor weights use shifted nonnegative scores so predictions are convex combinations of
    neighbor targets (stable for both cosine and Euclidean neighbor ranking).
    """

    def __init__(
        self,
        *,
        k: int = 50,
        metric: str = "cosine",
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        device: Optional[torch.device] = None,
        show_progress: bool = True,
    ):
        if k <= 0:
            raise ValueError("k must be positive.")
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be one of {'cosine','euclidean'}.")

        self.k = int(k)
        self.metric = metric
        self.encode_fn = encode_fn
        self.device = device if device is not None else torch.device("cpu")
        self.show_progress = bool(show_progress)

        self._train_embeddings: Optional[torch.Tensor] = None
        self._train_targets: Optional[torch.Tensor] = None
        self._train_sq_norms: Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit(self, train_loader: DataLoader) -> None:
        embeddings_list = []
        targets_list = []

        iterable = tqdm(train_loader, desc="kNN fit", leave=False) if self.show_progress else train_loader
        for batch_x, batch_y in iterable:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device, dtype=torch.float32)
            if batch_y.ndim != 2 or batch_y.size(-1) < 1:
                raise ValueError(
                    f"Expected batch_y shape (B, D) with D >= 1, got {tuple(batch_y.shape)}."
                )
            emb = self.encode_fn(batch_x)
            if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
                raise ValueError(
                    "encode_fn must return embeddings (B, D). "
                    f"Got type={type(emb)} shape={getattr(emb, 'shape', None)}."
                )
            emb = emb.to(dtype=torch.float32)
            if self.metric == "cosine":
                emb = F.normalize(emb, p=2, dim=1)
            embeddings_list.append(emb)
            targets_list.append(batch_y)

        self._train_embeddings = torch.cat(embeddings_list, dim=0).contiguous()
        self._train_targets = torch.cat(targets_list, dim=0).contiguous()
        if self.metric == "euclidean":
            self._train_sq_norms = (self._train_embeddings**2).sum(dim=1)

    @torch.no_grad()
    def _topk_neighbor_scores(self, test_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._train_embeddings is None or self._train_targets is None:
            raise RuntimeError("Call fit() before predicting.")

        x = test_embeddings.to(self.device, dtype=torch.float32)
        if self.metric == "cosine":
            x = F.normalize(x, p=2, dim=1)

        n_train = self._train_embeddings.size(0)
        k_eff = min(self.k, n_train)

        if self.metric == "cosine":
            sims = x @ self._train_embeddings.T
            return sims.topk(k_eff, dim=1, largest=True, sorted=True)

        if self._train_sq_norms is None:
            raise RuntimeError('Internal error: _train_sq_norms is None for metric="euclidean".')
        x_sq = (x**2).sum(dim=1, keepdim=True)
        dist_sq = x_sq + self._train_sq_norms.unsqueeze(0) - 2.0 * (x @ self._train_embeddings.T)
        scores = -dist_sq
        return scores.topk(k_eff, dim=1, largest=True, sorted=True)

    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor, input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        if self._train_targets is None:
            raise RuntimeError("Call fit() before predict_batch().")
        x = x.to(self.device)
        if input_transform is not None:
            x = input_transform(x)
        emb = self.encode_fn(x)
        if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
            raise ValueError(f"encode_fn must return (B, D), got {getattr(emb, 'shape', None)}")
        emb = emb.to(dtype=torch.float32)
        if self.metric == "cosine":
            emb = F.normalize(emb, p=2, dim=1)

        topk_indices, topk_scores = self._topk_neighbor_scores(emb)
        neighbor_y = self._train_targets[topk_indices.long()]
        # Non-negative weights => convex combination of neighbor targets (stable regression).
        # Raw cosine similarities can be negative; mixing signs in a weighted sum is not a proper average.
        score_min = topk_scores.min(dim=1, keepdim=True).values
        weights = (topk_scores - score_min) + 1e-12
        w = weights.unsqueeze(-1)
        pred = (neighbor_y * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-12)
        return pred

    @torch.no_grad()
    def test(
        self,
        test_loader: DataLoader,
        *,
        input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        denormalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Mean absolute error over all target dims and samples; RMSE = sqrt(mean ||pred - y||_2^2 per sample).

        If ``denormalize`` is set (maps targets to physical units, e.g. meters for distance), also returns
        ``mae_mean_meters`` and ``rmse_euclidean_meters`` on denormalized predictions and targets.
        """
        if self._train_embeddings is None:
            raise RuntimeError("Call fit() before test().")

        total_abs = 0.0
        total_dims = 0
        sse_l2 = 0.0
        n_samples = 0

        meter_abs = 0.0
        meter_dims = 0
        meter_sse = 0.0
        meter_n = 0

        iterable = tqdm(test_loader, desc="kNN test", leave=False) if self.show_progress else test_loader
        for batch_x, batch_y in iterable:
            batch_y = batch_y.to(self.device, dtype=torch.float32)
            pred = self.predict_batch(batch_x, input_transform=input_transform)

            diff = pred - batch_y
            total_abs += float(diff.abs().sum().item())
            total_dims += int(diff.numel())
            sse_l2 += float((diff**2).sum(dim=1).sum().item())
            n_samples += int(batch_y.size(0))

            if denormalize is not None:
                pred_m = denormalize(pred)
                y_m = denormalize(batch_y)
                dm = pred_m - y_m
                meter_abs += float(dm.abs().sum().item())
                meter_dims += int(dm.numel())
                meter_sse += float((dm**2).sum(dim=1).sum().item())
                meter_n += int(batch_y.size(0))

        out: Dict[str, float] = {
            "mae_mean": float(total_abs / max(total_dims, 1)),
            "rmse_euclidean": float((sse_l2 / max(n_samples, 1)) ** 0.5),
        }
        if denormalize is not None:
            out["mae_mean_meters"] = float(meter_abs / max(meter_dims, 1))
            out["rmse_euclidean_meters"] = float((meter_sse / max(meter_n, 1)) ** 0.5)
        return out
