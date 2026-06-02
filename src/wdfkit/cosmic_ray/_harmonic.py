# -*- coding: utf-8 -*-
"""Nd:YAG laser-harmonic notches (broad features, not cosmic-ray
spikes)."""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import xarray as xr

from .._shared._spectral import (
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)
from ._1d import linear_interpolate_masked_channels_1d

# Excitation laser in air; third harmonic of ~1064 nm Nd:YAG.
_ND_YAG_TRIGGER_NM_LOW = 354.0
_ND_YAG_TRIGGER_NM_HIGH = 356.0

# Catalogue lines (nm), WiRE-style.
_HARMONIC_WAVELENGTHS_NM: tuple[float, ...] = (1064.0, 532.0, 355.0, 266.0)

# ± nm around catalogue line to search for the broad peak top.
_SEARCH_HALF_WIDTH_NM = 2.5

# Total removal width in wavelength space (nm).
_NOTCH_FULL_WIDTH_NM = 1.0


def read_laser_wavelength_nm(attrs: Mapping[str, Any]) -> float | None:
    """Return ``laser_wavelength_nm`` from ``attrs``, or ``None``."""
    raw = attrs.get("laser_wavelength_nm")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _filename_for_messages(attrs: Mapping[str, Any]) -> str:
    return str(attrs.get("Filename") or "unknown file")


def should_apply_nd_yag_harmonic_cleanup(
    laser_wavelength_nm: float | None,
) -> bool:
    """True if excitation is the ~355 nm third-harmonic band."""
    if laser_wavelength_nm is None or not np.isfinite(laser_wavelength_nm):
        return False
    lo = _ND_YAG_TRIGGER_NM_LOW
    hi = _ND_YAG_TRIGGER_NM_HIGH
    return lo <= laser_wavelength_nm <= hi


def axis_is_absolute_wavenumber_cm(
    da: xr.DataArray,
    spectral_dim: str,
) -> bool:
    """True if the spectral coordinate is absolute wavenumber (cm⁻¹),
    not λ (nm).

    Raman *shift* axes are skipped (not absolute ν).
    """
    name_l = spectral_dim.lower()
    if name_l in ("raman_shift", "shifts"):
        return False
    if name_l == "nm":
        return False
    units = str(da[spectral_dim].attrs.get("units", "")).lower()
    if "nm" in units or "metre" in units or "meter" in units:
        return False
    if name_l == "wavenumber":
        return True
    return "cm^-1" in units or "1/cm" in units


def wavelength_nm_to_wavenumber_cm(wavelength_nm: float) -> float:
    """Cm⁻¹ from nm: ``1e7 / λ_nm`` (WiRE-style convention)."""
    return 1e7 / float(wavelength_nm)


def _axis_interval_around_wavelength_nm(
    harmonic_nm: float,
    *,
    wavenumber_axis: bool,
    half_width_nm: float,
) -> tuple[float, float]:
    """Closed search interval in spectral-axis units for
    ``harmonic_nm``."""
    lo_nm = harmonic_nm - half_width_nm
    hi_nm = harmonic_nm + half_width_nm
    if not wavenumber_axis:
        return (min(lo_nm, hi_nm), max(lo_nm, hi_nm))
    # Wavenumber decreases as wavelength increases.
    w_hi = wavelength_nm_to_wavenumber_cm(lo_nm) if lo_nm > 0 else np.inf
    w_lo = wavelength_nm_to_wavenumber_cm(hi_nm) if hi_nm > 0 else -np.inf
    return (min(w_lo, w_hi), max(w_lo, w_hi))


def _notch_interval_in_axis_units(
    peak_axis_value: float,
    *,
    wavenumber_axis: bool,
) -> tuple[float, float]:
    """±0.5 nm around peak, expressed as [low, high] in axis units."""
    half = _NOTCH_FULL_WIDTH_NM / 2.0
    if not wavenumber_axis:
        lo_nm = peak_axis_value - half
        hi_nm = peak_axis_value + half
        return (min(lo_nm, hi_nm), max(lo_nm, hi_nm))
    peak_nm = 1e7 / peak_axis_value
    lo_nm = peak_nm - half
    hi_nm = peak_nm + half
    if lo_nm <= 0 or hi_nm <= 0:
        return (peak_axis_value, peak_axis_value)
    w_hi = wavelength_nm_to_wavenumber_cm(lo_nm)
    w_lo = wavelength_nm_to_wavenumber_cm(hi_nm)
    return (min(w_lo, w_hi), max(w_lo, w_hi))


def _peak_wavelength_nm(peak_axis: float, wavenumber_axis: bool) -> float:
    if wavenumber_axis:
        return 1e7 / float(peak_axis)
    return float(peak_axis)


def _spectral_coord_in_interval_mask(
    x: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    a, b = (low, high) if low <= high else (high, low)
    return (x >= a) & (x <= b)


def remove_harmonic_notches_from_spectrum_1d(
    x_coord: np.ndarray,
    intensities: np.ndarray,
    *,
    wavenumber_axis: bool,
    filename: str,
) -> tuple[np.ndarray, list[float]]:
    """Notch ~1 nm in wavelength around each harmonic peak; linear
    interpolation.

    Prints one line per deleted peak.
    """
    if x_coord.size < 3 or intensities.shape != x_coord.shape:
        return intensities.copy(), []

    y_work = np.asarray(intensities, dtype=float).copy()
    x = np.asarray(x_coord, dtype=float)
    peaks_nm: list[float] = []

    for h_nm in _HARMONIC_WAVELENGTHS_NM:
        search_lo, search_hi = _axis_interval_around_wavelength_nm(
            h_nm,
            wavenumber_axis=wavenumber_axis,
            half_width_nm=_SEARCH_HALF_WIDTH_NM,
        )
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        if search_hi < x_min or search_lo > x_max:
            continue
        search_mask = _spectral_coord_in_interval_mask(x, search_lo, search_hi)
        if not np.any(search_mask):
            continue
        idx_candidates = np.flatnonzero(search_mask)
        local = y_work[search_mask]
        if local.size == 0:
            continue
        rel_imax = int(np.nanargmax(local))
        imax = int(idx_candidates[rel_imax])
        peak_x = float(x[imax])

        notch_lo, notch_hi = _notch_interval_in_axis_units(
            peak_x,
            wavenumber_axis=wavenumber_axis,
        )
        if notch_lo == notch_hi:
            continue
        remove_mask = _spectral_coord_in_interval_mask(x, notch_lo, notch_hi)
        if not np.any(remove_mask) or np.all(remove_mask):
            continue

        peak_nm = _peak_wavelength_nm(peak_x, wavenumber_axis)
        p = peak_nm
        msg = (
            f"355nm laser detected. Harmonic peak at {p:g} "
            f"nm is deleted for spectra {filename}"
        )
        print(msg)
        y_work = linear_interpolate_masked_channels_1d(y_work, remove_mask)
        peaks_nm.append(float(peak_nm))

    return y_work, peaks_nm


def harmonic_correct_dataarray(
    da: xr.DataArray,
    *,
    spectral_dim: str | None = None,
) -> xr.DataArray:
    """If ``laser_wavelength_nm`` is ~355 nm, notch laser harmonics on every
    slice.

    If any notch runs, merges ``treatments['Laser harmonic removal']``.
    Otherwise returns ``da`` unchanged (same object when no work done).
    """
    laser_nm = read_laser_wavelength_nm(da.attrs)
    if not should_apply_nd_yag_harmonic_cleanup(laser_nm):
        return da

    sdim = resolve_spectral_dim(da, spectral_dim)
    if axis_is_absolute_wavenumber_cm(da, sdim):
        wavenumber_axis = True
    else:
        units_s = str(da[sdim].attrs.get("units", "")).lower()
        if re.search(r"\bnm\b", units_s) or sdim.lower() == "nm":
            wavenumber_axis = False
        else:
            # Unknown axis (e.g. channel index): do not guess harmonics.
            return da

    filename = _filename_for_messages(da.attrs)
    da_t, orig_order = transpose_spectral_last(da, sdim)
    x = np.asarray(da_t[sdim].values, dtype=float)
    flat = da_t.values.reshape(-1, x.size)
    corrected = np.empty_like(flat, dtype=float)
    all_peaks: list[float] = []

    for row in range(flat.shape[0]):
        y_row, peaks = remove_harmonic_notches_from_spectrum_1d(
            x,
            flat[row],
            wavenumber_axis=wavenumber_axis,
            filename=filename,
        )
        corrected[row] = y_row
        all_peaks.extend(peaks)

    out_t = da_t.copy(data=corrected.reshape(da_t.shape))
    if out_t.dims != orig_order:
        out_t = out_t.transpose(*orig_order)

    if not all_peaks:
        return out_t

    meta = {
        "excitation_laser_nm": laser_nm,
        "harmonic_peaks_removed_nm": all_peaks,
    }
    return with_new_values(
        out_t,
        out_t.values,
        "Laser harmonic removal",
        meta,
    )
