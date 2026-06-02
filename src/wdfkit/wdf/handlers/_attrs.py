# -*- coding: utf-8 -*-
"""Shared ``make_attrs`` helper used by all handler modules."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import xarray as xr

from .._helpers.utils import hr_filesize

if TYPE_CHECKING:
    from ..parsed import ParsedWDF


def spectral_coord(parsed: "ParsedWDF") -> tuple:
    """Return an ``(dim_name, values, attrs)`` tuple for the spectral coord."""
    sdim = parsed.xlst.dim_name
    attrs: dict = {}
    if parsed.xlst.coord_units:
        attrs["units"] = parsed.xlst.coord_units
    attrs["long_name"] = parsed.xlst.units
    return sdim, parsed.xlst.values, attrs


def sort_spectral(da: "xr.DataArray") -> "xr.DataArray":
    """Sort *da* along its last (spectral) dimension in ascending order.

    This mirrors the ``da.sortby(sdim)`` call that the legacy assembler
    applied so that spectral values always run low→high.
    """
    sdim = da.dims[-1]
    coord = da[sdim].values
    if len(coord) > 1 and coord[0] > coord[-1]:
        da = da.isel({sdim: slice(None, None, -1)})
    return da


def make_attrs(parsed: "ParsedWDF", kind: str) -> dict:
    """Merge WDF1/WMAP params with the required new attrs for a DataArray.

    All existing param keys are preserved for backward compatibility.
    """
    attrs: dict = {}
    attrs.update(parsed.params)
    attrs.update(parsed.map_params)

    # Normalize WMAP InitialCoordinates from numpy array to dict
    if "InitialCoordinates" in attrs:
        ic = attrs["InitialCoordinates"]
        if not isinstance(ic, dict):
            attrs["InitialCoordinates"] = {
                "x": float(ic[0]),
                "y": float(ic[1]),
                "z": float(ic[2]),
            }

    # For scan types without a WMAP block (series, line_xy, points),
    # fall back to the WXIS stage position.
    if not attrs.get("InitialCoordinates"):
        attrs["InitialCoordinates"] = parsed.stage_xyz

    # --- genuinely new attrs (no existing equivalent in params) ---
    attrs["kind"] = kind
    attrs["spectral_units"] = parsed.xlst.units
    attrs["spectral_data_type"] = parsed.xlst.data_type
    if parsed.wmap is not None:
        attrs["wmap_flag"] = parsed.wmap.flag

    # --- file metadata ---
    attrs["Folder name"], attrs["Filename"] = os.path.split(parsed.filename)
    attrs["FileSize"] = hr_filesize(parsed.filesize)
    attrs["treatments"] = {}

    # --- optional extras ---
    if parsed.exposure_time is not None:
        attrs["ExposureTime"] = parsed.exposure_time
    if parsed.laser_power is not None:
        attrs["LaserPower"] = parsed.laser_power

    return attrs
