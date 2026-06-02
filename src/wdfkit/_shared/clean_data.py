# -*- coding: utf-8 -*-
"""Oversaturation detection and removal: :class:`CleanData`."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from ._spectral import resolve_spectral_dim

_TREATMENT_KEY = "Oversaturation Check"


def _extract_filename(da: xr.DataArray) -> str:
    for key in ("filename", "Title", "source"):
        val = da.attrs.get(key)
        if val:
            return str(val)
    return "unknown"


@dataclass
class CleanData:
    """Detect and remove oversaturated spectra (consecutive-zero runs).

    A spectrum is flagged when it contains at least ``n_zeros`` consecutive
    channels with value exactly zero — the signature of ADC saturation /
    detector clipping to zero.

    **1D**: emits a :class:`UserWarning` and returns the spectrum unchanged
    (the oversaturation is recorded in ``attrs["treatments"]``).

    **2D** ``(n_spectra, n_channels)``: removes flagged rows; returns a
    smaller DataArray with the same dimension names.  The removed coordinate
    values are recorded in ``attrs["treatments"]``.

    **3D** ``(ny, nx, n_channels)``: flattens spatial dims, removes flagged
    pixels, and returns a 2D DataArray ``(spectrum, n_channels)`` where
    ``spectrum`` is an integer index.  The original ``y`` / ``x`` coordinate
    values for each surviving spectrum are attached as non-dimension
    coordinates.  Removed pixel positions ``(y, x)`` are recorded in
    ``attrs["treatments"]``.

    Parameters
    ----------
    n_zeros
        Minimum length of a consecutive-zero run that flags a spectrum as
        oversaturated.  Default 10.
    spectral_dim
        Name of the spectral axis.  ``None`` → last dimension.
    """

    n_zeros: int = 10
    spectral_dim: str | None = None

    def __post_init__(self) -> None:
        if self.n_zeros < 1:
            raise ValueError("n_zeros must be >= 1")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, da: xr.DataArray) -> xr.DataArray:
        """Scan *da* for oversaturation; warn and/or remove bad spectra.

        Returns the original DataArray unchanged when no oversaturation is
        found, avoiding any copy overhead.
        """
        sdim = resolve_spectral_dim(da, self.spectral_dim)
        filename = _extract_filename(da)

        # Transpose so spectral is always last for consistent detection
        spatial_dims = [d for d in da.dims if d != sdim]
        da_sl = (
            da.transpose(*spatial_dims, sdim) if da.dims[-1] != sdim else da
        )

        arr = np.asarray(da_sl, dtype=float)
        flat = arr.reshape(-1, arr.shape[-1])
        bad_mask = self._find_bad_rows(flat)

        if not np.any(bad_mask):
            return da

        n_bad = int(bad_mask.sum())

        if da.ndim == 1:
            return self._handle_1d(da, filename)
        if da.ndim == 2:
            return self._handle_2d(
                da_sl, bad_mask, sdim, spatial_dims[0], filename, n_bad
            )
        if da.ndim == 3:
            return self._handle_3d(
                da_sl, bad_mask, sdim, spatial_dims, filename, n_bad
            )

        warnings.warn(
            f"Oversaturated spectra detected in '{filename}' but "
            f"CleanData does not support {da.ndim}-D input; skipping.",
            UserWarning,
            stacklevel=2,
        )
        return da

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _find_bad_rows(self, flat: np.ndarray) -> np.ndarray:
        """Return bool mask: True for rows with ``n_zeros``+ consecutive
        zeros."""
        n = self.n_zeros
        kernel = np.ones(n, dtype=np.int32)
        bad = np.zeros(flat.shape[0], dtype=bool)
        for i, row in enumerate(flat):
            is_zero = (row == 0.0).astype(np.int32)
            if is_zero.sum() < n:
                continue  # fast path: not enough zeros at all
            sums = np.convolve(is_zero, kernel, mode="valid")
            bad[i] = bool(np.any(sums >= n))
        return bad

    # ------------------------------------------------------------------
    # Per-dimensionality handlers
    # ------------------------------------------------------------------

    def _handle_1d(
        self,
        da: xr.DataArray,
        filename: str,
    ) -> xr.DataArray:
        warnings.warn(
            f"Oversaturated spectrum detected in '{filename}'.",
            UserWarning,
            stacklevel=3,
        )
        out = da.copy()
        meta: dict[str, Any] = {
            "detected": True,
            "action": "warning only — spectrum not modified",
            "filename": filename,
        }
        return _set_treatment(out, meta)

    def _handle_2d(
        self,
        da: xr.DataArray,
        bad_mask: np.ndarray,
        sdim: str,
        spatial_dim: str,
        filename: str,
        n_bad: int,
    ) -> xr.DataArray:
        good_idx = np.where(~bad_mask)[0]
        bad_idx = np.where(bad_mask)[0]

        if good_idx.size == 0:
            warnings.warn(
                f"All spectra in '{filename}' are oversaturated; "
                "returning original DataArray unchanged.",
                UserWarning,
                stacklevel=3,
            )
            return da

        # Coordinate values (or integer indices) of removed spectra
        if spatial_dim in da.coords:
            removed_coords = da.coords[spatial_dim].values[bad_idx].tolist()
        else:
            removed_coords = bad_idx.tolist()

        warnings.warn(
            f"Oversaturated spectra detected in '{filename}': "
            f"{n_bad} spectrum(a) removed along dim '{spatial_dim}'.",
            UserWarning,
            stacklevel=3,
        )

        da_clean = da.isel({spatial_dim: good_idx.tolist()})
        meta: dict[str, Any] = {
            "removed_count": n_bad,
            "dimension": spatial_dim,
            "removed_coords": {spatial_dim: removed_coords},
            "filename": filename,
        }
        return _set_treatment(da_clean, meta)

    def _handle_3d(
        self,
        da: xr.DataArray,
        bad_mask: np.ndarray,
        sdim: str,
        spatial_dims: list[str],
        filename: str,
        n_bad: int,
    ) -> xr.DataArray:
        y_dim, x_dim = spatial_dims[0], spatial_dims[1]
        ny, nx = da.shape[0], da.shape[1]
        bad_idx = np.where(bad_mask)[0]
        good_idx = np.where(~bad_mask)[0]

        if good_idx.size == 0:
            warnings.warn(
                f"All spectra in '{filename}' are oversaturated; "
                "returning original DataArray unchanged.",
                UserWarning,
                stacklevel=3,
            )
            return da

        # Coordinate arrays for spatial dims
        y_all = (
            da.coords[y_dim].values
            if y_dim in da.coords
            else np.arange(ny, dtype=float)
        )
        x_all = (
            da.coords[x_dim].values
            if x_dim in da.coords
            else np.arange(nx, dtype=float)
        )

        # (y, x) positions of removed pixels
        removed_positions = [
            {
                y_dim: float(y_all[i // nx]),
                x_dim: float(x_all[i % nx]),
            }
            for i in bad_idx.tolist()
        ]

        warnings.warn(
            f"Oversaturated spectra detected in '{filename}': "
            f"{n_bad} pixel(s) removed from map. "
            "Spatial dims flattened; result is 2D.",
            UserWarning,
            stacklevel=3,
        )

        # Build 2D output: (n_valid, n_channels)
        flat = da.values.reshape(-1, da.shape[-1])
        clean_flat = flat[good_idx]

        spec_coords = (
            da.coords[sdim].values
            if sdim in da.coords
            else np.arange(da.shape[-1])
        )
        y_good = y_all[good_idx // nx]
        x_good = x_all[good_idx % nx]

        da_clean = xr.DataArray(
            clean_flat,
            dims=("spectrum", sdim),
            coords={
                "spectrum": np.arange(good_idx.size),
                sdim: spec_coords,
                y_dim: ("spectrum", y_good),
                x_dim: ("spectrum", x_good),
            },
            attrs=dict(da.attrs),
        )
        meta: dict[str, Any] = {
            "removed_count": n_bad,
            "y_dim": y_dim,
            "x_dim": x_dim,
            "removed_positions": removed_positions,
            "filename": filename,
        }
        return _set_treatment(da_clean, meta)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _set_treatment(
    da: xr.DataArray,
    meta: dict[str, Any],
) -> xr.DataArray:
    """Write *meta* into ``da.attrs["treatments"][_TREATMENT_KEY]``."""
    treats = dict(da.attrs.get("treatments") or {})
    treats[_TREATMENT_KEY] = meta
    da.attrs["treatments"] = treats
    return da


__all__ = ["CleanData"]
