"""
Task-agnostic kNN evaluator for classification.

This implements a simple kNN classifier that:
1) `fit(train_loader)`: extracts embeddings and stores them along with integer class labels.
2) `test(test_loader)`: extracts embeddings for the test set, predicts class ids, and reports accuracy.
3) `test_topk(test_loader, max_k, input_transform=None)`: top-1..top-k classification accuracy from
   neighbor vote scores; optional `input_transform` on inputs before encoding (e.g. noise).

The evaluator is task-agnostic: it only assumes that the dataloaders yield `(x, y)` where
`y` are integer class ids, and that the `encode_fn` maps `x` to `(B, D)` embeddings.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _infer_num_classes(labels: torch.Tensor) -> int:
    if labels.numel() == 0:
        return 0
    max_label = int(labels.max().item())
    min_label = int(labels.min().item())
    if min_label != 0:
        raise ValueError(
            f"Labels are expected to start at 0 for classification. Got min_label={min_label}."
        )
    return max_label + 1


class kNNforClassification:
    """
    kNN classifier for embeddings extracted from a pretrained model.

    Parameters
    ----------
    k:
        Number of nearest neighbors.
    metric:
        Similarity metric used for neighbor search:
        - "cosine": cosine similarity (dot-product after L2-normalization)
        - "euclidean": negative squared Euclidean distance (higher is better)
    encode_fn:
        Function mapping `x` to embeddings of shape (B, D).
        This is required; the evaluator does not assume that the dataloader
        already yields embeddings.
    device:
        Device for similarity computation.
    show_progress:
        If True, wrap `fit` and `test` loops with a tqdm progress bar.
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

        self._train_embeddings: Optional[torch.Tensor] = None  # (N_train, D)
        self._train_labels: Optional[torch.Tensor] = None  # (N_train,)
        self._num_classes: Optional[int] = None  # (N_train,)

        # Precomputed ||y||^2 for each train embedding y, used to speed up
        # squared-distance computation for metric="euclidean".
        self._train_sq_norms: Optional[torch.Tensor] = None  # (N_train,)
        
    @torch.no_grad()
    def fit(self, train_loader: DataLoader) -> None:
        """
        Fit by caching train embeddings and their labels.
        """
        embeddings_list = []
        labels_list = []

        iterable = tqdm(train_loader, desc="kNN fit", leave=False) if self.show_progress else train_loader
        for batch_x, batch_y in iterable:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device, dtype=torch.int64)
            emb = self.encode_fn(batch_x)

            if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
                raise ValueError(
                    "encode_fn (or loader x) must produce embeddings with shape (B, D). "
                    f"Got emb type={type(emb)} shape={getattr(emb, 'shape', None)}."
                )

            emb = emb.to(dtype=torch.float32)
            if self.metric == "cosine":
                # L2-normalize to turn cosine similarity into a dot-product:
                #   cosine(a,b) = (a/||a||) · (b/||b||)
                emb = F.normalize(emb, p=2, dim=1)

            embeddings_list.append(emb)
            labels_list.append(batch_y)

        # `.contiguous()` ensures a compact memory layout after concatenation.
        # This helps keep later similarity/indexing ops efficient.
        self._train_embeddings = torch.cat(embeddings_list, dim=0).contiguous()
        self._train_labels = torch.cat(labels_list, dim=0).contiguous()
        self._num_classes = _infer_num_classes(self._train_labels)

        if self.metric == "euclidean":
            self._train_sq_norms = (self._train_embeddings**2).sum(dim=1)

    @torch.no_grad()
    def _topk_neighbor_scores(
        self, test_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return top-k neighbor indices for each test embedding.

        Returns
        -------
        topk_indices: LongTensor of shape (B, k)
        """
        if self._train_embeddings is None or self._train_labels is None:
            raise RuntimeError("Call fit() before predicting.")

        x = test_embeddings.to(self.device, dtype=torch.float32)
        if self.metric == "cosine":
            # Same normalization as in fit() for consistent cosine similarity.
            x = F.normalize(x, p=2, dim=1)

        # Fast-path for GPU: compute full (B, N_train) similarity/dist matrix once
        # and take top-k neighbors directly.
        n_train = self._train_embeddings.size(0)
        k = min(self.k, n_train)

        if self.metric == "cosine":
            # Similarity (B, C): dot-product of L2-normalized vectors.
            sims = x @ self._train_embeddings.T
            topk_scores, topk_indices = sims.topk(k, dim=1, largest=True, sorted=True)
            return topk_indices, topk_scores

        # Metric == "euclidean": use negative squared distance as "higher is better".
        # dist^2(x,y) = ||x||^2 + ||y||^2 - 2 x·y
        if self._train_sq_norms is None:
            raise RuntimeError('Internal error: _train_sq_norms is None for metric="euclidean".')

        # x_sq[b] = ||x_b||^2, shaped (B, 1) for broadcasting.
        x_sq = (x**2).sum(dim=1, keepdim=True)  # (B, 1)

        # dist_sq[b,c] = ||x_b - y_c||^2 for all train embeddings y_c.
        #   - x_sq: (B,1)  -> ||x_b||^2
        #   - _train_sq_norms: (C,) -> ||y_c||^2 (broadcasted to (1,C))
        #   - x @ train.T: (B,C) -> x_b · y_c
        dist_sq = (
            x_sq
            + self._train_sq_norms.unsqueeze(0)
            - 2.0 * (x @ self._train_embeddings.T)
        )  # (B, C)

        # Convert distance to a score where "larger is better" for top-k.
        # With topk(largest=True), nearest neighbors correspond to the
        # highest scores = -dist_sq.
        scores = -dist_sq
        topk_scores, topk_indices = scores.topk(k, dim=1, largest=True, sorted=True)
        return topk_indices, topk_scores

    @torch.no_grad()
    def _weighted_votes_from_embeddings(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Aggregate neighbor votes into a per-class score matrix of shape (B, num_classes).
        """
        if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
            raise ValueError(
                "embeddings must be a tensor of shape (B, D). "
                f"Got shape={getattr(emb, 'shape', None)}."
            )
        emb = emb.to(self.device, dtype=torch.float32)
        if self.metric == "cosine":
            emb = F.normalize(emb, p=2, dim=1)

        topk_indices, topk_scores = self._topk_neighbor_scores(emb)
        assert self._train_labels is not None
        neighbor_labels = self._train_labels[topk_indices]  # (B, k_nn)

        assert self._num_classes is not None

        weights = topk_scores
        if self.metric == "euclidean":
            score_min = topk_scores.min(dim=1, keepdim=True).values
            weights = (topk_scores - score_min) + 1e-12

        vote = torch.zeros(
            (emb.size(0), self._num_classes),
            device=self.device,
            dtype=torch.float32,
        )
        vote.scatter_add_(dim=1, index=neighbor_labels, src=weights)
        return vote

    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class ids for a batch.
        """
        emb = self.encode_fn(x)
        vote = self._weighted_votes_from_embeddings(emb)
        return vote.argmax(dim=1)

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate accuracy on `test_loader`.
        """
        if self._train_embeddings is None:
            raise RuntimeError("Call fit() before test().")

        correct = 0
        total = 0

        iterable = tqdm(test_loader, desc="kNN test", leave=False) if self.show_progress else test_loader
        for batch_x, batch_y in iterable:
            batch_y = batch_y.to(self.device, dtype=torch.int64)
            preds = self.predict_batch(batch_x)
            correct += int((preds == batch_y).sum().item())
            total += int(batch_y.numel())

        acc = correct / max(1, total)
        return {"accuracy": float(acc)}

    @torch.no_grad()
    def test_topk(
        self,
        test_loader: DataLoader,
        *,
        max_k: int = 5,
        input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Classification top-k accuracy: fraction of samples whose true label is among
        the k classes with highest kNN vote totals (k = 1 .. max_k).

        Parameters
        ----------
        max_k:
            Largest k to report (e.g. 5 -> top_1, ..., top_5).
        input_transform:
            Optional transform applied to batch_x on device before encoding
            (e.g. add noise for robustness evaluation).
        """
        if self._train_embeddings is None:
            raise RuntimeError("Call fit() before test_topk().")
        if max_k <= 0:
            raise ValueError("max_k must be positive.")

        correct: List[int] = [0] * max_k
        totals: List[int] = [0] * max_k

        iterable = (
            tqdm(test_loader, desc="kNN test top-k", leave=False)
            if self.show_progress
            else test_loader
        )
        for batch_x, batch_y in iterable:
            x = batch_x.to(self.device)
            if input_transform is not None:
                x = input_transform(x)
            batch_y = batch_y.to(self.device, dtype=torch.int64)

            emb = self.encode_fn(x)
            vote = self._weighted_votes_from_embeddings(emb)

            num_classes = vote.size(1)
            k_cap = min(max_k, num_classes)
            top_class_idx = vote.topk(k_cap, dim=1, largest=True, sorted=True).indices
            y_col = batch_y.unsqueeze(1)

            for ki in range(1, max_k + 1):
                kk = min(ki, k_cap)
                hits = (top_class_idx[:, :kk] == y_col).any(dim=1)
                correct[ki - 1] += int(hits.sum().item())
                totals[ki - 1] += int(batch_y.numel())

        out: Dict[str, float] = {}
        for ki in range(1, max_k + 1):
            out[f"top_{ki}"] = float(correct[ki - 1] / max(1, totals[ki - 1]))
        return out

