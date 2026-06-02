# -*- coding: utf-8 -*-
"""High-level per-spectrum smoothing: :class:`SpectraSmoother`.

Applies Savitzky-Golay or Whittaker-Eilers filtering independently to every
spectrum in a DataArray of any shape.  Works on 1-D single spectra, 2-D
stacks, and 3-D map cubes — the spectral axis is always the last dimension
after an internal transpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr

from .._shared._spectral import (
    reshape_row_stack_to,
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)
from ..wdf._helpers.utils import ensure_in_memory
from ._smooth_1d import auto_lam, savgol_smooth_1d, whittaker_smooth_1d

SmoothMethod = Literal["savgol", "whittaker"]

_TREATMENT_KEY = "spectra_smoothing"


@dataclass
class SpectraSmoother:
    """Per-spectrum 1-D smoothing for DataArrays of any shape.

    Applies the chosen filter independently to every spectrum along the
    spectral axis.  Suitable for 1-D single spectra, 2-D stacks
    ``(n_spectra, spectral)``, and 3-D map cubes ``(y, x, spectral)``.

    Parameters
    ----------
    method
        ``\"savgol\"`` (default) — Savitzky-Golay filter via
        ``scipy.signal.savgol_filter``.
        ``\"whittaker\"`` — Whittaker-Eilers smoother (sparse linear system).
    window_length
        Savitzky-Golay: number of channels in the filter window.
        Must be odd and >= ``polyorder + 2``.
    polyorder
        Savitzky-Golay: polynomial order (must be < ``window_length``).
    lam
        Whittaker-Eilers: smoothness penalty λ.  ``None`` (default) triggers
        automatic selection via GCV minimisation (see ``auto_lam_calls``).
    d
        Whittaker-Eilers: difference order (default 2).
    auto_lam_calls
        Maximum GCV evaluations when ``lam=None`` (default 5).
    spectral_dim
        Name of the spectral axis.  Defaults to the last dimension.
    """

    method: SmoothMethod = "savgol"
    window_length: int = 11
    polyorder: int = 3
    lam: float | None = None
    d: int = 2
    auto_lam_calls: int = 5
    spectral_dim: str | None = None

    def __post_init__(self) -> None:
        allowed: tuple[str, ...] = ("savgol", "whittaker")
        if self.method not in allowed:
            raise ValueError(
                f"method must be one of {allowed!r}; got {self.method!r}"
            )
        if self.window_length % 2 == 0 or self.window_length < 3:
            raise ValueError(
                f"window_length must be odd and >= 3; got {self.window_length}"
            )
        if self.polyorder >= self.window_length:
            raise ValueError(
                f"polyorder ({self.polyorder}) must be < "
                f"window_length ({self.window_length})"
            )
        if self.lam is not None and self.lam <= 0:
            raise ValueError(f"lam must be positive; got {self.lam}")
        if self.d < 1:
            raise ValueError(f"d must be >= 1; got {self.d}")
        if self.auto_lam_calls < 1:
            raise ValueError(
                f"auto_lam_calls must be >= 1; got {self.auto_lam_calls}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def smooth(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Return a smoothed copy of ``spectrum``.

        Works on DataArrays of any shape.  Each spectrum (row along the
        spectral axis) is smoothed independently.
        """
        if not isinstance(spectrum, xr.DataArray):
            raise TypeError(
                "SpectraSmoother.smooth expects an xarray.DataArray; got "
                f"{type(spectrum).__name__}"
            )
        cleaned, meta = self._smooth_core(spectrum)
        return with_new_values(spectrum, cleaned, _TREATMENT_KEY, meta)

    def transform(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Alias of :meth:`smooth`."""
        return self.smooth(spectrum)

    # ------------------------------------------------------------------
    # Internal helpers (also used by SpectraCleaner)
    # ------------------------------------------------------------------

    def _smooth_core(
        self, spectra: xr.DataArray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Validate, load, smooth, and return ``(cleaned_array, meta)``.

        The returned array has the same shape as ``spectra.values``.
        """
        sdim = resolve_spectral_dim(spectra, self.spectral_dim)
        da_w, orig_order = transpose_spectral_last(spectra, sdim)

        da_w = ensure_in_memory(
            da_w,
            caller="SpectraSmoother",
            reason="Per-spectrum smoothing requires the full array in memory.",
            stacklevel=3,
        )

        values = da_w.values
        original_shape = values.shape
        n_spectral = original_shape[-1]

        # Flatten to (n_spectra, n_spectral) — works for any ndim ≥ 1
        if values.ndim == 1:
            row_stack = values.reshape(1, n_spectral)
        else:
            row_stack = values.reshape(-1, n_spectral)

        cleaned_rows, meta = self._smooth_rows(row_stack)
        meta["spectral_dim"] = sdim

        cleaned_w = reshape_row_stack_to(cleaned_rows, original_shape)

        # Restore original dimension order if we transposed
        if tuple(da_w.dims) != orig_order:
            cleaned_da = da_w.copy(data=cleaned_w).transpose(*orig_order)
            cleaned = cleaned_da.values
        else:
            cleaned = cleaned_w

        return cleaned, meta

    def _smooth_rows(
        self, row_stack: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply the smoother to every row of ``(n_spectra, n_spectral)``.

        Returns ``(cleaned_rows, meta)``.  When ``lam=None`` for the
        Whittaker method the optimal λ is determined once from the mean
        spectrum and reused for all rows.
        """
        n_spectra, n_spectral = row_stack.shape
        cleaned = np.empty_like(row_stack, dtype=np.float64)
        meta: dict[str, Any] = {"method": self.method}

        if self.method == "savgol":
            meta["window_length"] = self.window_length
            meta["polyorder"] = self.polyorder
            for i in range(n_spectra):
                cleaned[i] = savgol_smooth_1d(
                    row_stack[i], self.window_length, self.polyorder
                )

        else:  # whittaker
            meta["d"] = self.d
            if self.lam is not None:
                lam_used = self.lam
                meta["lam"] = lam_used
                meta["lam_auto"] = False
            else:
                # Estimate λ once from the mean spectrum
                mean_spec = row_stack.mean(axis=0)
                lam_used = auto_lam(
                    mean_spec, d=self.d, max_calls=self.auto_lam_calls
                )
                meta["lam_used"] = lam_used
                meta["lam_auto"] = True
                meta["auto_lam_calls"] = self.auto_lam_calls

            for i in range(n_spectra):
                cleaned[i] = whittaker_smooth_1d(
                    row_stack[i], lam_used, self.d
                )

        meta["n_spectra"] = n_spectra
        meta["n_spectral"] = n_spectral
        return cleaned, meta


__all__ = ["SpectraSmoother", "SmoothMethod"]
