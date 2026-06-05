# -*- coding: utf-8 -*-
"""Handler for ``kind="points"`` (flag == 0x01, RandomPoints)."""

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
    """Return a 2-D DataArray ``("point", "spectral")`` with x/y coords."""
    nspectra = parsed.nspectra
    data = np.asarray(parsed.data)

    point_idx = np.arange(nspectra)
    sc = spectral_coord(parsed)
    coords: dict = {
        "point": point_idx,
        "spectral": ("spectral", sc[1], sc[2]),
    }
    orgn_x = parsed.orgn_by_type("SpatialX")
    orgn_y = parsed.orgn_by_type("SpatialY")
    if orgn_x is not None:
        coords["x"] = ("point", orgn_x.values[:nspectra])
    if orgn_y is not None:
        coords["y"] = ("point", orgn_y.values[:nspectra])

    attrs = make_attrs(parsed, "points")
    attrs["shape"] = (nspectra, 1)
    attrs["data_type"] = "sequence"

    return sort_spectral(
        xr.DataArray(
            data,
            dims=("point", "spectral"),
            coords=coords,
            attrs=attrs,
        )
    )


class PointsHandler(ScanHandler):
    kind = "points"

    def matches(self, parsed: "ParsedWDF") -> bool:
        return (
            parsed.measurement_type == MeasurementType.Map
            and parsed.wmap is not None
            and parsed.wmap.flag == MapFlag.RandomPoints
        )

    def build(self, parsed: "ParsedWDF") -> xr.DataArray:
        return build_dataarray(parsed)
