# -*- coding: utf-8 -*-
"""Handler for ``kind="volume"`` (nsteps[2] > 1, 3-D spatial grid)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 4-D DataArray ``(z, y, x, spectral_dim)``.

    .. note::
        Not yet supported. Raises :exc:`NotImplementedError`.
    """
    raise NotImplementedError(
        "Volume scans (kind='volume', 4-D output z/y/x/spectral) are not "
        "supported yet. Use a raster or series scan instead."
    )
    sdim = parsed.xlst.dim_name
    assert parsed.wmap is not None, "volume requires a WMAP block"
    nx = int(parsed.wmap.nsteps[0])
    ny = int(parsed.wmap.nsteps[1])
    nz = int(parsed.wmap.nsteps[2])
    npts = parsed.npoints

    data = np.asarray(parsed.data)
    cube = data.reshape(nz, ny, nx, npts)

    sc = spectral_coord(parsed)
    coords: dict = {sdim: (sdim, sc[1], sc[2])}
    orgn_x = parsed.orgn_by_type("SpatialX")
    orgn_y = parsed.orgn_by_type("SpatialY")
    orgn_z = parsed.orgn_by_type("SpatialZ")
    nsp = nz * ny * nx
    if orgn_x is not None:
        x_3d = orgn_x.values[:nsp].reshape(nz, ny, nx)
        coords["x"] = ("x", x_3d[0, 0, :])
    if orgn_y is not None:
        y_3d = orgn_y.values[:nsp].reshape(nz, ny, nx)
        coords["y"] = ("y", y_3d[0, :, 0])
    if orgn_z is not None:
        z_3d = orgn_z.values[:nsp].reshape(nz, ny, nx)
        coords["z"] = ("z", z_3d[:, 0, 0])

    attrs = make_attrs(parsed, "volume")
    attrs["shape"] = (nz, ny, nx)

    return sort_spectral(
        xr.DataArray(
            cube,
            dims=("z", "y", "x", sdim),
            coords=coords,
            attrs=attrs,
        )
    )
