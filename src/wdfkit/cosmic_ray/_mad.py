# -*- coding: utf-8 -*-
"""Robust noise scale (MAD) helpers for cosmic-ray detection."""

from __future__ import annotations

import numpy as np

# Scales MAD to a nominal Gaussian standard deviation for thresholding.
_SCALED_MAD_TO_GAUSSIAN_SIGMA = 1.4826


def scaled_median_absolute_deviation_noise(x: np.ndarray) -> float:
    """``1.4826 * median(|x - median(x)|)`` — robust spread of ``x``."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return _SCALED_MAD_TO_GAUSSIAN_SIGMA * mad


def noise_estimate_too_small(
    noise: float,
    reference_scale: float,
) -> bool:
    """True if ``noise`` is too small or non-finite for stable
    thresholding."""
    if not np.isfinite(noise):
        return True
    floor = 1e-15 * (reference_scale + np.finfo(float).tiny)
    return noise <= 0.0 or noise < floor


def robust_mad_noise_with_floor(
    deviations: np.ndarray,
    amplitude_reference: float,
) -> float:
    """Scaled MAD of ``deviations``; if tiny, bump using
    ``amplitude_reference``."""
    noise = scaled_median_absolute_deviation_noise(deviations)
    ref = float(amplitude_reference) + np.finfo(float).tiny
    if noise < 1e-15 * ref:
        noise = max(noise, 1e-12 * ref)
    return noise
