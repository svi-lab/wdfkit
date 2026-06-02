# -*- coding: utf-8 -*-
"""Handler for ``kind="raster_rowmajor"`` (flag == 0x00, StandardRaster)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 3-D DataArray ``(y, x, spectral_dim)``.

    Data is acquired row-by-row (Y-outer, X-inner), so a direct reshape
    from acquisition order to ``(ny, nx, npts)`` is correct.
    """
    sdim = parsed.xlst.dim_name
    assert parsed.wmap is not None, "raster_rowmajor requires a WMAP block"
    nx, ny = int(parsed.wmap.nsteps[0]), int(parsed.wmap.nsteps[1])
    npts = parsed.npoints

    data = np.asarray(parsed.data)
    cube = data.reshape(ny, nx, npts)

    # Extract 1-D axis vectors from ORGN spatial coords.
    sc = spectral_coord(parsed)
    coords: dict = {sdim: (sdim, sc[1], sc[2])}
    orgn_x = parsed.orgn_by_type("SpatialX")
    orgn_y = parsed.orgn_by_type("SpatialY")
    if orgn_x is not None:
        x_2d = orgn_x.values[: ny * nx].reshape(ny, nx)
        coords["x"] = ("x", x_2d[0, :])
    if orgn_y is not None:
        y_2d = orgn_y.values[: ny * nx].reshape(ny, nx)
        coords["y"] = ("y", y_2d[:, 0])

    attrs = make_attrs(parsed, "raster_rowmajor")
    attrs["shape"] = (ny, nx)

    return sort_spectral(
        xr.DataArray(
            cube,
            dims=("y", "x", sdim),
            coords=coords,
            attrs=attrs,
        )
    )
