# -*- coding: utf-8 -*-
"""1D spectrum cosmic-ray (positive spike) removal."""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

from ._mad import noise_estimate_too_small, robust_mad_noise_with_floor


def _coerce_float_1d_spectrum(y: np.ndarray, kernel_size: int) -> np.ndarray:
    """Cast ``y`` to float 1D; validate ``kernel_size`` is odd and ≥ 3."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {arr.shape}")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"spike_width must be odd and >= 3, got {kernel_size}"
        )
    return arr


def _dilate_mask_1d(mask: np.ndarray) -> np.ndarray:
    """Expand a boolean mask by 1 channel on each side."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    return out


def _zero_saturation_mask(y: np.ndarray) -> np.ndarray:
    """Flag near-zero channels that are surrounded by positive neighbours.

    Detects detector-saturation artifacts where the ADC clips to 0 instead
    of returning the true count.  A channel ``i`` is flagged when:

    * ``y[i]`` is below ``1e-4 × median(positive values)``, AND
    * at least 2 of the 4 nearest neighbours exceed 10% of that median.
    """
    pos = y[y > 0]
    if pos.size < 3:
        return np.zeros(y.size, dtype=bool)
    pos_median = float(np.median(pos))
    floor = 1e-4 * pos_median
    nbr_thr = 0.1 * pos_median
    near_zero = y <= floor
    is_pos = (y > nbr_thr).astype(np.int8)
    padded = np.pad(is_pos, 2, mode="edge")
    nbr_sum = padded[:-4] + padded[1:-3] + padded[3:-1] + padded[4:]
    return near_zero & (nbr_sum >= 2)


def positive_spike_mask_vs_median_smooth(
    y: np.ndarray,
    median_smoothed_y: np.ndarray,
    threshold_multiplier: float,
) -> tuple[np.ndarray, float]:
    """Mask where positive residual exceeds ``threshold_multiplier *
    noise``.

    Residual is ``y - median_smoothed_y``; ``noise`` is scaled MAD of residual.
    """
    residual = y - median_smoothed_y
    if not np.any(residual):
        return np.zeros(y.shape, dtype=bool), 0.0
    amplitude_reference = max(
        float(np.nanmax(np.abs(y))),
        float(np.nanmax(np.abs(median_smoothed_y))),
    )
    noise = robust_mad_noise_with_floor(
        residual,
        amplitude_reference,
    )
    mask = residual > threshold_multiplier * noise
    return mask.astype(bool), noise


def positive_spike_mask_from_derivative_peaks(
    y: np.ndarray,
    threshold_multiplier: float,
) -> np.ndarray:
    """Interior ``i`` where ``y[i]`` is above both neighbors by
    ``threshold_multiplier * noise``.

    ``noise`` is scaled MAD of ``diff(y)``.
    """
    dy = np.diff(y)
    n = y.size
    mask = np.zeros(n, dtype=bool)
    if dy.size == 0:
        return mask
    amplitude_reference = max(
        float(np.nanmax(np.abs(y))),
        float(np.nanmax(np.abs(dy))),
    )
    noise = robust_mad_noise_with_floor(dy, amplitude_reference)
    max_abs_dy = float(np.nanmax(np.abs(dy))) + np.finfo(float).tiny
    if noise_estimate_too_small(noise, max_abs_dy):
        return mask
    threshold = threshold_multiplier * noise
    for i in range(1, n - 1):
        if (y[i] - y[i - 1] > threshold) and (y[i] - y[i + 1] > threshold):
            mask[i] = True
    return mask


def linear_interpolate_masked_channels_1d(
    y: np.ndarray,
    bad_channel_mask: np.ndarray,
) -> np.ndarray:
    """Fill masked channels by linear interpolation from good ones."""
    if not np.any(bad_channel_mask):
        return y.copy()
    good = ~bad_channel_mask
    if not np.any(good):
        return y.copy()
    n = y.size
    x = np.arange(n, dtype=float)
    out = y.copy()
    bad_idx = np.flatnonzero(bad_channel_mask)
    good_idx = np.flatnonzero(good)
    if good_idx.size == 1:
        out[bad_idx] = out[good_idx[0]]
        return out
    out[bad_idx] = np.interp(x[bad_idx], x[good_idx], y[good_idx])
    return out


def remove_cosmic_rays_1d(
    y: np.ndarray,
    *,
    kernel_size: int = 5,
    threshold: float = 5.0,
    max_passes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove sharp positive spikes from one 1D spectrum.

    Uses a ``scipy.signal.medfilt`` reference and MAD-based noise estimation.
    Operates on the raw counts / intensity array (only masked indices change).

    The algorithm runs up to ``max_passes`` iterations.  Each pass:

    1. Detects new spikes on the current (already-repaired) signal.
    2. **Dilates** the new spike mask by 1 channel on each side to catch
       sub-threshold spike edges.
    3. Accumulates into a single cumulative mask across all passes.
    4. **Repairs** by linear interpolation from the *original* signal at all
       cumulative masked positions — avoids chaining interpolation errors.

    Early termination when a pass finds no new spikes.

    Parameters
    ----------
    y
        One spectral trace (any numeric dtype; cast to float).
    kernel_size
        Odd length ``>= 3`` for ``medfilt``.  Increase for broader spikes
        (e.g. ``9``–``13`` for 7–10 channel-wide cosmic rays).
    threshold
        Multiplier on MAD-derived noise (larger → fewer detections).
    max_passes
        Maximum number of detection–repair iterations (default 3).

    Returns
    -------
    corrected_y
        Same shape as ``y``; unchanged if no spikes found or noise degenerate.
    cosmic_mask
        Boolean mask, same shape as ``y``; ``True`` at all corrected channels.
    """
    y1 = _coerce_float_1d_spectrum(y, kernel_size)
    n = y1.size
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be positive and finite")
    if max_passes < 1:
        raise ValueError("max_passes must be >= 1")

    # Saturated-zero detection runs once on original signal
    zero_mask = _zero_saturation_mask(y1)
    cumulative_mask = zero_mask.copy()
    current = (
        linear_interpolate_masked_channels_1d(y1, zero_mask)
        if np.any(zero_mask)
        else y1.copy()
    )

    for _ in range(max_passes):
        median_filtered = medfilt(current, kernel_size=kernel_size)
        new_mask, _ = positive_spike_mask_vs_median_smooth(
            current, median_filtered, threshold
        )

        if not np.any(new_mask):
            break

        new_mask = _dilate_mask_1d(new_mask)
        cumulative_mask |= new_mask

        if np.all(cumulative_mask):
            return y1.copy(), np.zeros(n, dtype=bool)

        current = linear_interpolate_masked_channels_1d(y1, cumulative_mask)

    return current, cumulative_mask
