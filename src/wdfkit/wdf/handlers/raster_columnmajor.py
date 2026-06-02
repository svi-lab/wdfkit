# -*- coding: utf-8 -*-
"""Handler for ``kind="raster_columnmajor"`` (flag == 0x02, StreamLine).

StreamLine acquires column-by-column (X-outer, Y-inner), so the flat
acquisition order is (X0Y0, X0Y1, ..., X0Yn-1, X1Y0, ...).  We reshape
to ``(nx, ny, npts)`` then transpose to ``(ny, nx, npts)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 3-D DataArray ``(y, x, spectral_dim)``."""
    sdim = parsed.xlst.dim_name
    assert parsed.wmap is not None, "raster_columnmajor requires a WMAP block"
    nx, ny = int(parsed.wmap.nsteps[0]), int(parsed.wmap.nsteps[1])
    npts = parsed.npoints

    data = np.asarray(parsed.data)
    # Acquisition order: X-outer, Y-inner → reshape to (nx, ny, npts)
    cube = data.reshape(nx, ny, npts).transpose(1, 0, 2)  # → (ny, nx, npts)

    sc = spectral_coord(parsed)
    coords: dict = {sdim: (sdim, sc[1], sc[2])}
    orgn_x = parsed.orgn_by_type("SpatialX")
    orgn_y = parsed.orgn_by_type("SpatialY")
    if orgn_x is not None:
        # After reshape(nx, ny): row i → col i, col j → row j
        x_nx_ny = orgn_x.values[: nx * ny].reshape(nx, ny)
        # Transpose to (ny, nx): [row_j, col_i] → x values are along axis 1
        coords["x"] = ("x", x_nx_ny.T[0, :])
    if orgn_y is not None:
        y_nx_ny = orgn_y.values[: nx * ny].reshape(nx, ny)
        coords["y"] = ("y", y_nx_ny.T[:, 0])

    attrs = make_attrs(parsed, "raster_columnmajor")
    attrs["ScanShape"] = (ny, nx)
    attrs["RowCoord"] = "y"
    attrs["ColCoord"] = "x"

    return sort_spectral(
        xr.DataArray(
            cube,
            dims=("y", "x", sdim),
            coords=coords,
            attrs=attrs,
        )
    )
