# -*- coding: utf-8 -*-
"""Handler for ``kind="series"`` (MeasurementType == 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 2-D DataArray ``(series_axis, spectral_dim)``.

    The series axis is the first ORGN entry with the primary-axis flag set.
    Falls back to the first non-flag / non-checksum ORGN entry, then to a
    plain integer index if nothing is found.
    """
    sdim = parsed.xlst.dim_name
    nspectra = parsed.ncollected or parsed.nspectra
    data = np.asarray(parsed.data)[:nspectra]

    # Locate the primary series axis
    primary = parsed.primary_orgn()
    if primary is None:
        # Fall back to the first double-typed ORGN entry that isn't Time/Flags
        for entry in parsed.orgn:
            if entry.data_type not in (
                "Checksum",
                "Flags",
                "Time",
                "Arbitrary",
            ):
                primary = entry
                break
    if primary is None:
        # Last resort: use integer index
        row_dim = "index"
        row_values = np.arange(nspectra)
        row_units = ""
    else:
        row_dim = primary.data_type  # e.g. "SpatialZ"
        row_values = primary.values[:nspectra]
        row_units = primary.units

    attrs = make_attrs(parsed, "series")
    attrs["ScanShape"] = (nspectra, 1)
    attrs["RowCoord"] = row_dim
    attrs["ColCoord"] = None
    if row_units:
        attrs["units_axis"] = row_units

    sc = spectral_coord(parsed)
    da = xr.DataArray(
        data,
        dims=(row_dim, sdim),
        coords={
            row_dim: row_values,
            sdim: (sdim, sc[1], sc[2]),
        },
        attrs=attrs,
    )
    return sort_spectral(da)
