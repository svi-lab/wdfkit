# -*- coding: utf-8 -*-
"""Handler for ``kind="raster_columnmajor"`` (flag == 0x02, ColumnMajor).

ColumnMajor acquires column-by-column (X-outer, Y-inner), so the flat
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

from ..types import MapFlag, MeasurementType
from .base import ScanHandler


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 3-D DataArray ``("row", "column", "spectral")``.

    row = y axis (physical µm), column = x axis (physical µm).
    """
    assert parsed.wmap is not None, "raster_columnmajor requires a WMAP block"
    nx, ny = int(parsed.wmap.nsteps[0]), int(parsed.wmap.nsteps[1])
    npts = parsed.npoints

    data = np.asarray(parsed.data)
    # Acquisition order: X-outer, Y-inner → reshape to (nx, ny, npts)
    cube = data.reshape(nx, ny, npts).transpose(1, 0, 2)  # → (ny, nx, npts)

    sc = spectral_coord(parsed)
    coords: dict = {"spectral": ("spectral", sc[1], sc[2])}
    orgn_x = parsed.orgn_by_type("SpatialX")
    orgn_y = parsed.orgn_by_type("SpatialY")
    if orgn_x is not None:
        x_nx_ny = orgn_x.values[: nx * ny].reshape(nx, ny)
        coords["column"] = ("column", x_nx_ny.T[0, :])
    if orgn_y is not None:
        y_nx_ny = orgn_y.values[: nx * ny].reshape(nx, ny)
        coords["row"] = ("row", y_nx_ny.T[:, 0])
    time_entry = parsed.orgn_by_type("ElapsedTime") or parsed.orgn_by_type(
        "Time"
    )
    if time_entry is not None:
        coords["time"] = (
            ("row", "column"),
            time_entry.values[: ny * nx].reshape(ny, nx),
        )

    attrs = make_attrs(parsed, "raster_columnmajor")
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


class RasterColumnMajorHandler(ScanHandler):
    kind = "raster_columnmajor"

    def matches(self, parsed: "ParsedWDF") -> bool:
        return (
            parsed.measurement_type == MeasurementType.Map
            and parsed.wmap is not None
            and parsed.wmap.flag == MapFlag.ColumnMajor
            and int(parsed.wmap.nsteps[2]) == 1
        )

    def build(self, parsed: "ParsedWDF") -> xr.DataArray:
        return build_dataarray(parsed)
