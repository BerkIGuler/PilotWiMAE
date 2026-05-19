"""Channel prediction baselines (linear interpolation, Kronecker LMMSE).

NMSE in baseline JSON outputs uses ``metric_version`` 2 fields: **non-pilot** time–frequency
REs only (all antennas), **one** linear NMSE per full ``[T, N_a, N_f]`` channel tensor,
then mean across samples (and optionally std across folds). Full-grid ensemble NMSE from the
TeX document is not reported as the headline metric. Specification:
``context/LMMSE_implementation/lmmse_baseline.tex``.
"""

from .evaluate import evaluate_linear_interpolation
from .linear_interp import (
    fill_frame_linear_frequency,
    interpolate_time_linear,
    reconstruct_linear_from_pilots,
)
from .lmmse_core import (
    compute_lmmse_weights_W,
    estimate_R_t_R_sf,
    kronecker_blocks,
    lmmse_estimate_channel,
    prepare_correlations,
    vec_time_ant_freq,
)
from .masks import (
    expanded_subcarrier_indices,
    non_pilot_mask_from_pilots,
    pilot_time_indices,
    validate_index_bounds,
)
from .metrics import (
    mse,
    nmse,
    nmse_masked,
    nmse_non_pilot_tf,
    nmse_to_db,
)
from .npz_io import (
    build_sorted_npz_list,
    compute_dataset_mean_complex_power,
    compute_pref_reference_power,
    iter_channels,
)
from .pilot_noise import (
    apply_pilot_awgn,
    make_observed_from_target,
    parse_pilot_pattern,
)

__all__ = [
    "apply_pilot_awgn",
    "build_sorted_npz_list",
    "compute_dataset_mean_complex_power",
    "compute_lmmse_weights_W",
    "compute_pref_reference_power",
    "estimate_R_t_R_sf",
    "evaluate_linear_interpolation",
    "expanded_subcarrier_indices",
    "fill_frame_linear_frequency",
    "interpolate_time_linear",
    "iter_channels",
    "kronecker_blocks",
    "lmmse_estimate_channel",
    "make_observed_from_target",
    "mse",
    "nmse",
    "nmse_masked",
    "nmse_non_pilot_tf",
    "nmse_to_db",
    "non_pilot_mask_from_pilots",
    "parse_pilot_pattern",
    "pilot_time_indices",
    "prepare_correlations",
    "reconstruct_linear_from_pilots",
    "validate_index_bounds",
    "vec_time_ant_freq",
]
