# -*- coding: utf-8 -*-
"""Per-spectrum normalization (dynamic spectral coordinate)."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from sklearn import preprocessing

from ..wdf._helpers.utils import ensure_in_memory
from ._spectral import (
    reshape_row_stack_to,
    resolve_spectral_dim,
    transpose_spectral_last,
    with_new_values,
)

# Methods that normalize each spectrum using only its own values.
# These are fully parallelisable and work chunk-by-chunk via apply_ufunc.
_PER_SPECTRUM_METHODS = {"l1", "l2", "max", "min_max", "area"}

# Methods that require statistics across ALL spectra simultaneously.
# A Dask-backed DataArray will be computed fully before these run.
_GLOBAL_METHODS = {"wave_number", "robust_scale"}


def _trapz_y(
    y: np.ndarray,
    x: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Trapezoidal integration along ``axis``.

    Prefer ``numpy.trapezoid`` (NumPy 2+); fall back to ``numpy.trapz``.
    Avoid ``getattr(..., np.trapz)``: the default is evaluated eagerly and
    ``np.trapz`` is absent on some NumPy 2 builds.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def _normalize_numpy_block(
    spectra_2d: np.ndarray,
    method: str,
    x_values: np.ndarray,
    kwargs: dict[str, Any],
) -> np.ndarray:
    """Normalize a 2D ``(n_spectra, n_points)`` NumPy array.

    This is the pure-NumPy inner kernel, called directly for ndarray input
    and as the ``apply_ufunc`` kernel for per-spectrum methods on DataArrays.
    """
    if method in ("l1", "l2", "max"):
        out = preprocessing.normalize(
            spectra_2d, axis=1, norm=method, copy=False
        )
    elif method == "min_max":
        out = preprocessing.minmax_scale(spectra_2d, axis=1, copy=False)
    elif method == "area":
        denom = _trapz_y(spectra_2d, x_values, axis=-1)[:, np.newaxis]
        denom = np.where(np.abs(denom) < np.finfo(float).eps, 1.0, denom)
        out = spectra_2d / denom
    elif method == "wave_number":
        wave_number = kwargs.get("wave_number", float(np.min(x_values)))
        idx = int(np.nanargmin(np.abs(x_values - wave_number)))
        divider = spectra_2d[:, [idx]]
        mean_divider = float(np.mean(divider))
        divider = np.where(divider == 0, mean_divider, divider)
        out = spectra_2d / divider
    elif method == "robust_scale":
        quantile = kwargs.get("quantile", (5.0, 95.0))
        centering = kwargs.get("centering", False)
        out = preprocessing.robust_scale(
            spectra_2d,
            axis=1,
            with_centering=centering,
            quantile_range=quantile,
        )
    else:
        raise ValueError(
            f"normalize method {method!r} is not recognised. "
            '"method" must be one of '
            '["l1", "l2", "max", "min_max", "wave_number", '
            '"robust_scale", "area"].'
        )

    out = out - np.min(out, axis=-1, keepdims=True)
    return out


def _make_apply_ufunc_kernel(
    method: str,
    x_values: np.ndarray,
    kwargs: dict[str, Any],
):
    """Return a function ``(spectra_nd,) → normalized_nd`` for use with
    ``xr.apply_ufunc``."""

    def _kernel(arr: np.ndarray) -> np.ndarray:
        orig_shape = arr.shape
        arr_2d = arr.reshape(-1, orig_shape[-1])
        out_2d = _normalize_numpy_block(arr_2d, method, x_values, kwargs)
        return out_2d.reshape(orig_shape)

    return _kernel


_ALL_METHODS = _PER_SPECTRUM_METHODS | _GLOBAL_METHODS


def normalize(
    input_spectra: xr.DataArray | np.ndarray,
    method: str = "robust_scale",
    *,
    spectral_dim: str | None = None,
    **kwargs,
) -> xr.DataArray | np.ndarray:
    """Scale spectra along the spectral axis.

    For :class:`xarray.DataArray` input, the spectral axis defaults to the
    **last** dimension (e.g. ``nm``, ``raman_shift``, ``shifts``, …). Pass
    ``spectral_dim`` to select another dimension when spectra are not last.

    **Dask-backed DataArrays** are handled transparently:

    - *Per-spectrum methods* (``"l1"``, ``"l2"``, ``"max"``, ``"min_max"``,
      ``"area"``): processed chunk-by-chunk via ``xr.apply_ufunc`` —
      no data is loaded into RAM beyond the current chunk.
    - *Global methods* (``"robust_scale"``, ``"wave_number"``): require
      statistics across all spectra; the full array is computed first.  A
      ``UserWarning`` is emitted so you know RAM is being used.

    Parameters
    ----------
    input_spectra
        DataArray or 2D ndarray of shape ``(n_spectra, n_points)``.
    method
        One of ``"l1"``, ``"l2"``, ``"max"``, ``"min_max"``,
        ``"wave_number"``, ``"robust_scale"``, ``"area"``.
    spectral_dim
        Spectral dimension name when ``input_spectra`` is a DataArray.
    x_values
        Spectral abscissa for ndarray input (default ``arange(n_points)``).

    Returns
    -------
    Same type as ``input_spectra`` with updated ``attrs["treatments"]`` for
    DataArray output.
    """
    if method not in _ALL_METHODS:
        raise ValueError(
            f"normalize method {method!r} is not recognised. "
            f'"method" must be one of {sorted(_ALL_METHODS)!r}.'
        )

    if isinstance(input_spectra, xr.DataArray):
        return _normalize_dataarray(
            input_spectra, method, spectral_dim, kwargs
        )

    # --- ndarray path ---
    spectra = np.asarray(input_spectra)
    if spectra.ndim != 2:
        raise ValueError(
            "ndarray input must be 2D with shape (n_spectra, n_points)"
        )
    x_values = kwargs.get("x_values")
    if x_values is None:
        x_values = np.arange(spectra.shape[-1])
    else:
        x_values = np.asarray(x_values)
    return _normalize_numpy_block(spectra, method, x_values, kwargs).reshape(
        spectra.shape
    )


def _normalize_dataarray(
    da: xr.DataArray,
    method: str,
    spectral_dim: str | None,
    kwargs: dict[str, Any],
) -> xr.DataArray:
    """DataArray normalisation, Dask-aware."""
    sdim = resolve_spectral_dim(da, spectral_dim)
    da_w, orig_order = transpose_spectral_last(da, sdim)
    x_values = da_w[sdim].values

    meta_keys = (
        "quantile",
        "centering",
        "wave_number",
        "x_values",
        "spectral_dim",
    )
    meta = {k: kwargs[k] for k in meta_keys if k in kwargs}
    treatment_payload = {"method": method, **meta}

    is_dask = da_w.chunks is not None

    if is_dask and method in _GLOBAL_METHODS:
        da_w = ensure_in_memory(
            da_w,
            caller=f'normalize(method="{method}")',
            reason=(
                "Requires statistics across all spectra and cannot be "
                "computed chunk-by-chunk. For memory-efficient normalisation "
                f"use a per-spectrum method "
                f"({', '.join(sorted(_PER_SPECTRUM_METHODS))})."
            ),
            stacklevel=3,
        )
        is_dask = False

    if is_dask:
        kernel = _make_apply_ufunc_kernel(method, x_values, kwargs)
        out_w = xr.apply_ufunc(
            kernel,
            da_w,
            input_core_dims=[[sdim]],
            output_core_dims=[[sdim]],
            dask="parallelized",
            output_dtypes=[da_w.dtype],
        )
    else:
        spectra_2d = da_w.values.reshape(-1, da_w.shape[-1])
        out_2d = _normalize_numpy_block(spectra_2d, method, x_values, kwargs)
        packed = reshape_row_stack_to(out_2d, da_w.shape)
        out_w = da_w.copy(data=packed)

    if tuple(out_w.dims) != orig_order:
        out_w = out_w.transpose(*orig_order)

    return with_new_values(da, out_w.data, "normalization", treatment_payload)
