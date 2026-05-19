"""
Beam prediction data utilities.

This package hosts shared beam task components (codebooks, label generation,
and related helpers) that are useful across downstream tasks.
"""

from .beam_codebook import (
    generate_upa_2d_dft_codebook,
    num_beams_from_saved_codebook,
    upa_2d_dft_num_beams,
    upa_axis_dft_codewords,
)
from .channel_noise import add_complex_awgn_snr_db
from .datasets import BeamLabelDatasetWrapper, collate_beam_targets

__all__ = [
    "add_complex_awgn_snr_db",
    "BeamLabelDatasetWrapper",
    "collate_beam_targets",
    "generate_upa_2d_dft_codebook",
    "num_beams_from_saved_codebook",
    "upa_2d_dft_num_beams",
    "upa_axis_dft_codewords",
]

