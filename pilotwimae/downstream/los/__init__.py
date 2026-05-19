"""
LoS vs. nLoS classification on embeddings (kNN CLI: ``evaluate_knn``).
"""

from pilotwimae.downstream.los.datasets import LosBinaryLabelDataset, load_los_binary_labels

__all__ = ["LosBinaryLabelDataset", "load_los_binary_labels"]
