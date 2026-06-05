# -*- coding: utf-8 -*-
"""Handler for ``kind="raster_rowmajor"`` (flag == 0x00, StandardRaster)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF

import warnings

from ..types import MapFlag, MeasurementType
from .base import ScanHandler


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 3-D DataArray ``("row", "column", "spectral")``.

    Data is acquired row-by-row (Y-outer, X-inner), so a direct reshape
    from acquisition order to ``(ny, nx, npts)`` is correct.
    row = y axis (physical µm), column = x axis (physical µm).
    """
    assert parsed.wmap is not None, "raster_rowmajor requires a WMAP block"
    nx, ny = int(parsed.wmap.nsteps[0]), int(parsed.wmap.nsteps[1])
    npts = parsed.npoints

    data = np.asarray(parsed.data)
    cube = data.reshape(ny, nx, npts)

    sc = spectral_coord(parsed)
    coords: dict = {"spectral": ("spectral", sc[1], sc[2])}
    orgn_x = parsed.orgn_by_type("SpatialX")
    orgn_y = parsed.orgn_by_type("SpatialY")
    if orgn_x is not None:
        x_2d = orgn_x.values[: ny * nx].reshape(ny, nx)
        coords["column"] = ("column", x_2d[0, :])
    if orgn_y is not None:
        y_2d = orgn_y.values[: ny * nx].reshape(ny, nx)
        coords["row"] = ("row", y_2d[:, 0])
    time_entry = parsed.orgn_by_type("ElapsedTime") or parsed.orgn_by_type(
        "Time"
    )
    if time_entry is not None:
        coords["time"] = (
            ("row", "column"),
            time_entry.values[: ny * nx].reshape(ny, nx),
        )

    attrs = make_attrs(parsed, "raster_rowmajor")
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


class RasterRowMajorHandler(ScanHandler):
    """Catch-all for Map scans not matched by any other handler.

    Matches any remaining Map scan (StandardRaster or truly unknown flag).
    Emits a warning for unknown flag combinations.
    """

    kind = "raster_rowmajor"

    def matches(self, parsed: "ParsedWDF") -> bool:
        return (
            parsed.measurement_type == MeasurementType.Map
            and parsed.wmap is not None
        )

    def build(self, parsed: "ParsedWDF") -> xr.DataArray:
        flag = parsed.wmap.flag
        nsteps = parsed.wmap.nsteps
        if flag != MapFlag.StandardRaster or nsteps[2] != 1:
            warnings.warn(
                f"Unknown WMAP flag={flag:#x}/nsteps={nsteps!r} combination; "
                "falling back to 'raster_rowmajor'",
                UserWarning,
                stacklevel=4,
            )
        return build_dataarray(parsed)
