# -*- coding: utf-8 -*-
"""Handler for ``kind="line_xy"`` (flag == 0x80, XYLine)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 2-D DataArray ``(point, spectral_dim)`` with x/y coords."""
    sdim = parsed.xlst.dim_name
    nspectra = parsed.nspectra
    data = np.asarray(parsed.data)

    point_idx = np.arange(nspectra)
    sc = spectral_coord(parsed)
    coords: dict = {
        "point": point_idx,
        sdim: (sdim, sc[1], sc[2]),
    }
    orgn_x = parsed.orgn_by_type("SpatialX")
    orgn_y = parsed.orgn_by_type("SpatialY")
    if orgn_x is not None:
        coords["x"] = ("point", orgn_x.values[:nspectra])
    if orgn_y is not None:
        coords["y"] = ("point", orgn_y.values[:nspectra])

    attrs = make_attrs(parsed, "line_xy")
    attrs["shape"] = (nspectra, 1)

    return sort_spectral(
        xr.DataArray(
            data,
            dims=("point", sdim),
            coords=coords,
            attrs=attrs,
        )
    )
