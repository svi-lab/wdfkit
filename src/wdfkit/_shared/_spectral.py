# -*- coding: utf-8 -*-
"""Shared helpers for preprocessing DataArrays."""

from __future__ import annotations

import numpy as np
import xarray as xr


def resolve_spectral_dim(
    da: xr.DataArray,
    spectral_dim: str | None,
) -> str:
    """Return the spectral dimension name (last dim by default)."""
    if spectral_dim is not None:
        if spectral_dim not in da.dims:
            raise ValueError(
                f"spectral_dim {spectral_dim!r} is not among dims {da.dims!r}"
            )
        return spectral_dim
    if not da.dims:
        raise ValueError("DataArray has no dimensions")
    return da.dims[-1]


def transpose_spectral_last(
    da: xr.DataArray,
    spectral_dim: str,
) -> tuple[xr.DataArray, tuple[str, ...]]:
    """Return ``(dataarray, original_dim_order)`` with spectral axis last."""
    order = (*[d for d in da.dims if d != spectral_dim], spectral_dim)
    orig = tuple(da.dims)
    if order == orig:
        return da, orig
    return da.transpose(*order), orig


def reshape_row_stack_to(
    stacked_spectra: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Reshape ``(n_spectra, n_channels)`` back to ``target_shape``."""
    return stacked_spectra.reshape(target_shape)


def with_new_values(
    template: xr.DataArray,
    values: np.ndarray,
    treatment_key: str,
    treatment_value: dict,
) -> xr.DataArray:
    """Copy ``template`` with new ``values`` and merged ``treatments``."""
    out = template.copy(data=values)
    treats = dict(out.attrs.get("treatments") or {})  # fixed: was "Treatments"
    prev = treats.get(treatment_key, {})
    treats[treatment_key] = {**prev, **treatment_value}
    out.attrs["treatments"] = treats
    return out
