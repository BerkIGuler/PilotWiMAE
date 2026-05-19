"""
Shared type aliases specific to beam prediction downstream task.
"""

from typing import Literal


LabelMode = Literal["sequence", "snapshot"]
ReturnFormat = Literal["class_index", "topk_indices", "indices_and_scores"]
