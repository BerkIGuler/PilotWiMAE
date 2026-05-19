"""Linear interpolation utilities for channel-prediction baselines."""

from __future__ import annotations

import numpy as np

from .masks import validate_index_bounds


def _interp_vector(values: np.ndarray, known_indices: np.ndarray, *, outside_mode: str) -> np.ndarray:
    """Linearly interpolate one 1D vector with configurable edge behavior."""
    n = int(values.shape[0])
    if n == 0:
        return values
    idx = np.array(sorted({int(i) for i in known_indices.tolist()}), dtype=np.int64)
    if idx.size == 0:
        raise ValueError("known_indices must not be empty")
    if idx[0] < 0 or idx[-1] >= n:
        raise ValueError("known_indices out of bounds for vector length")
    if idx.size == 1:
        return np.full_like(values, values[idx[0]])
    x = np.arange(n, dtype=np.float64)
    known_vals = values[idx]
    out = np.interp(x, idx.astype(np.float64), known_vals)
    mode = str(outside_mode).strip().lower()
    if mode not in {"hold", "linear"}:
        raise ValueError("frequency_outside_mode must be one of {'hold', 'linear'}")
    if mode == "linear":
        left_mask = x < float(idx[0])
        right_mask = x > float(idx[-1])
        if np.any(left_mask):
            m_left = (known_vals[1] - known_vals[0]) / float(idx[1] - idx[0])
            out[left_mask] = known_vals[0] + m_left * (x[left_mask] - float(idx[0]))
        if np.any(right_mask):
            m_right = (known_vals[-1] - known_vals[-2]) / float(idx[-1] - idx[-2])
            out[right_mask] = known_vals[-1] + m_right * (x[right_mask] - float(idx[-1]))
    return out.astype(values.dtype, copy=False)


def fill_frame_linear_frequency(
    frame_2d: np.ndarray,
    *,
    known_subcarriers: list[int],
    frequency_axis: int = -1,
    frequency_outside_mode: str = "hold",
) -> np.ndarray:
    """
    Fill one frame by linear interpolation along the frequency axis.

    Parameters
    ----------
    frame_2d:
        2D grid (e.g. antennas x subcarriers or subcarriers x antennas).
    known_subcarriers:
        Subcarrier indices that are known from pilots.
    frequency_axis:
        Axis containing frequency/subcarrier index in ``frame_2d``.
    """
    arr = np.asarray(frame_2d)
    if arr.ndim != 2:
        raise ValueError(f"frame_2d must be 2D, got shape {arr.shape}")
    if np.iscomplexobj(arr):
        real = fill_frame_linear_frequency(
            arr.real,
            known_subcarriers=known_subcarriers,
            frequency_axis=frequency_axis,
            frequency_outside_mode=frequency_outside_mode,
        )
        imag = fill_frame_linear_frequency(
            arr.imag,
            known_subcarriers=known_subcarriers,
            frequency_axis=frequency_axis,
            frequency_outside_mode=frequency_outside_mode,
        )
        return real + 1j * imag

    f_axis = frequency_axis if frequency_axis >= 0 else arr.ndim + frequency_axis
    if f_axis not in (0, 1):
        raise ValueError(f"frequency_axis must be 0 or 1 for 2D frame, got {frequency_axis}")
    validate_index_bounds(known_subcarriers, upper_bound=arr.shape[f_axis], name="known_subcarriers")
    known = np.array(sorted(set(int(i) for i in known_subcarriers)), dtype=np.int64)

    moved = np.moveaxis(arr, f_axis, -1)
    out = np.empty_like(moved)
    for i in range(moved.shape[0]):
        out[i] = _interp_vector(moved[i], known, outside_mode=frequency_outside_mode)
    return np.moveaxis(out, -1, f_axis)


def interpolate_time_linear(
    time0: np.ndarray,
    time1: np.ndarray,
    *,
    t0: int,
    t1: int,
    t_query: int,
) -> np.ndarray:
    """Linear interpolation between two frames at times t0 and t1."""
    if int(t1) == int(t0):
        raise ValueError("t0 and t1 must differ for interpolation")
    a = (float(t_query) - float(t0)) / (float(t1) - float(t0))
    return (1.0 - a) * np.asarray(time0) + a * np.asarray(time1)


def reconstruct_linear_from_pilots(
    grid: np.ndarray,
    *,
    pilot_times: list[int],
    known_subcarriers: list[int],
    frequency_axis: int = -1,
    time_axis: int = 0,
    frequency_outside_mode: str = "hold",
    time_outside_mode: str = "hold",
) -> np.ndarray:
    """
    Reconstruct full grid using linear interpolation within frame and across time.

    ``grid`` shape is expected to include a time axis and a frequency axis.
    Any leading dimensions are treated as batch-like and preserved.
    """
    arr = np.asarray(grid)
    if arr.ndim < 3:
        raise ValueError(f"grid must be at least 3D (time + frame dims), got {arr.shape}")
    t_axis = time_axis if time_axis >= 0 else arr.ndim + time_axis
    f_axis = frequency_axis if frequency_axis >= 0 else arr.ndim + frequency_axis
    if t_axis == f_axis:
        raise ValueError("time_axis and frequency_axis must differ")

    validate_index_bounds(pilot_times, upper_bound=arr.shape[t_axis], name="pilot_times")
    pilot_times_sorted = sorted(set(int(t) for t in pilot_times))
    if len(pilot_times_sorted) != 2:
        raise ValueError("This baseline currently requires exactly two pilot times")
    t0, t1 = pilot_times_sorted
    mode = str(time_outside_mode).strip().lower()
    if mode not in {"hold", "linear"}:
        raise ValueError("time_outside_mode must be one of {'hold', 'linear'}")
    f_mode = str(frequency_outside_mode).strip().lower()
    if f_mode not in {"hold", "linear"}:
        raise ValueError("frequency_outside_mode must be one of {'hold', 'linear'}")

    moved = np.moveaxis(arr, (t_axis, f_axis), (0, -1))
    # moved shape: (T, *rest, F)
    t_len = moved.shape[0]
    rest_shape = moved.shape[1:-1]
    f_len = moved.shape[-1]
    validate_index_bounds(known_subcarriers, upper_bound=f_len, name="known_subcarriers")

    flat = moved.reshape(t_len, int(np.prod(rest_shape, dtype=np.int64)), f_len)
    out = flat.copy()

    out[t0] = np.stack(
        [
            _interp_vector(flat[t0, i], np.asarray(known_subcarriers), outside_mode=f_mode)
            for i in range(flat.shape[1])
        ],
        axis=0,
    )
    out[t1] = np.stack(
        [
            _interp_vector(flat[t1, i], np.asarray(known_subcarriers), outside_mode=f_mode)
            for i in range(flat.shape[1])
        ],
        axis=0,
    )

    for t in range(t_len):
        if t == t0 or t == t1:
            continue
        if t < t0:
            if mode == "hold":
                out[t] = out[t0]
            else:
                out[t] = interpolate_time_linear(out[t0], out[t1], t0=t0, t1=t1, t_query=t)
        elif t > t1:
            if mode == "hold":
                out[t] = out[t1]
            else:
                out[t] = interpolate_time_linear(out[t0], out[t1], t0=t0, t1=t1, t_query=t)
        else:
            out[t] = interpolate_time_linear(out[t0], out[t1], t0=t0, t1=t1, t_query=t)

    rebuilt = out.reshape((t_len,) + rest_shape + (f_len,))
    return np.moveaxis(rebuilt, (0, -1), (t_axis, f_axis))
