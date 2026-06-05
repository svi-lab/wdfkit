# -*- coding: utf-8 -*-
"""Handler for ``kind="raster_snake"`` (flag & 0x04, Alternating/serpentine).

Odd rows (1, 3, 5, …) are acquired right-to-left; we reverse them so that
all rows share the same X ordering (left-to-right).

If ``nsteps`` reports ``(1, 1, 1)`` (irregular / non-rectangular snake),
the data cannot be reshaped into a 2-D grid; we fall back to point-list
representation with ``dims=(point, spectral_dim)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF

from ..types import MapFlag, MeasurementType
from .base import ScanHandler


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return ``("row", "column", "spectral")`` or ``("point", "spectral")``
    DataArray depending on whether a rectangular grid can be recovered."""
    assert parsed.wmap is not None, "raster_snake requires a WMAP block"
    nx, ny = int(parsed.wmap.nsteps[0]), int(parsed.wmap.nsteps[1])
    npts = parsed.npoints
    nspectra = parsed.nspectra

    data = np.asarray(parsed.data)
    sc = spectral_coord(parsed)

    # Irregular snake (circle fill etc.): cannot reshape to rect grid.
    if nx * ny != nspectra or nx <= 1 or ny <= 1:
        coords: dict = {
            "point": np.arange(nspectra),
            "spectral": ("spectral", sc[1], sc[2]),
        }
        orgn_x = parsed.orgn_by_type("SpatialX")
        orgn_y = parsed.orgn_by_type("SpatialY")
        if orgn_x is not None:
            coords["x"] = ("point", orgn_x.values[:nspectra])
        if orgn_y is not None:
            coords["y"] = ("point", orgn_y.values[:nspectra])
        attrs = make_attrs(parsed, "raster_snake")
        attrs["shape"] = (nspectra, 1)
        attrs["data_type"] = "sequence"
        return sort_spectral(
            xr.DataArray(
                data, dims=("point", "spectral"), coords=coords, attrs=attrs
            )
        )

    # Rectangular snake: reshape then un-serpentine.
    cube = data.reshape(ny, nx, npts).copy()
    cube[1::2] = cube[1::2, ::-1]  # reverse odd rows

    coords = {"spectral": ("spectral", sc[1], sc[2])}
    orgn_x = parsed.orgn_by_type("SpatialX")
    orgn_y = parsed.orgn_by_type("SpatialY")
    if orgn_x is not None:
        x_2d = orgn_x.values[: ny * nx].reshape(ny, nx).copy()
        x_2d[1::2] = x_2d[1::2, ::-1]
        coords["column"] = ("column", x_2d[0, :])
    if orgn_y is not None:
        y_2d = orgn_y.values[: ny * nx].reshape(ny, nx).copy()
        y_2d[1::2] = y_2d[1::2, ::-1]
        coords["row"] = ("row", y_2d[:, 0])
    time_entry = parsed.orgn_by_type("ElapsedTime") or parsed.orgn_by_type(
        "Time"
    )
    if time_entry is not None:
        coords["time"] = (
            ("row", "column"),
            time_entry.values[: ny * nx].reshape(ny, nx),
        )

    attrs = make_attrs(parsed, "raster_snake")
    attrs["shape"] = (ny, nx)
    attrs["data_type"] = "grid"
    attrs["row_axis"] = "y"
    attrs["column_axis"] = "x"

    return sort_spectral(
        xr.DataArray(
            cube,
            dims=("row", "column", "spectral"),
            coords=coords,
            attrs=attrs,
        )
    )


class RasterSnakeHandler(ScanHandler):
    kind = "raster_snake"

    def matches(self, parsed: "ParsedWDF") -> bool:
        return (
            parsed.measurement_type == MeasurementType.Map
            and parsed.wmap is not None
            and bool(parsed.wmap.flag & MapFlag.Alternating)
        )

    def build(self, parsed: "ParsedWDF") -> xr.DataArray:
        return build_dataarray(parsed)
