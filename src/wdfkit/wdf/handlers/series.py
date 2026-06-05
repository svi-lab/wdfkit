# -*- coding: utf-8 -*-
"""Handler for ``kind="series"`` (MeasurementType == 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF

from ..types import MeasurementType
from .base import ScanHandler

_ORGN_COORD: dict[str, str] = {
    "SpatialX": "x",
    "SpatialY": "y",
    "SpatialZ": "z",
}


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 2-D DataArray ``("point", "spectral")``.

    Spatial ORGN entries (SpatialX/Y/Z) become coords along the point dim.
    A time coord is added when an ElapsedTime or Time ORGN entry is present.
    """
    nspectra = parsed.ncollected or parsed.nspectra
    data = np.asarray(parsed.data)[:nspectra]
    sc = spectral_coord(parsed)

    coords: dict = {
        "point": np.arange(nspectra),
        "spectral": ("spectral", sc[1], sc[2]),
    }
    for entry in parsed.orgn:
        coord_name = _ORGN_COORD.get(entry.data_type)
        if coord_name:
            coords[coord_name] = ("point", entry.values[:nspectra])

    time_entry = parsed.orgn_by_type("ElapsedTime") or parsed.orgn_by_type(
        "Time"
    )
    if time_entry is not None:
        coords["time"] = ("point", time_entry.values[:nspectra])

    attrs = make_attrs(parsed, "series")
    attrs["shape"] = (nspectra, 1)
    attrs["data_type"] = "sequence"

    da = xr.DataArray(
        data,
        dims=("point", "spectral"),
        coords=coords,
        attrs=attrs,
    )
    return sort_spectral(da)


class SeriesHandler(ScanHandler):
    kind = "series"

    def matches(self, parsed: "ParsedWDF") -> bool:
        return parsed.measurement_type == MeasurementType.Series

    def build(self, parsed: "ParsedWDF") -> xr.DataArray:
        return build_dataarray(parsed)
