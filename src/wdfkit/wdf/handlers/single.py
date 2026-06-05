# -*- coding: utf-8 -*-
"""Handler for ``kind="single"`` (MeasurementType == 1, nspectra == 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF

from .base import ScanHandler


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 1-D DataArray ``("spectral",)``."""
    data = np.asarray(parsed.data)
    if data.ndim == 2:
        spectrum = data[0]
    else:
        spectrum = data

    attrs = make_attrs(parsed, "single")
    attrs["shape"] = (1, 1)
    attrs["data_type"] = "single"

    sc = spectral_coord(parsed)
    da = xr.DataArray(
        spectrum,
        dims=("spectral",),
        coords={"spectral": ("spectral", sc[1], sc[2])},
        attrs=attrs,
    )
    return sort_spectral(da)


class SingleHandler(ScanHandler):
    kind = "single"

    def matches(self, parsed: "ParsedWDF") -> bool:
        return True  # absolute catch-all; must be last in the registry

    def build(self, parsed: "ParsedWDF") -> xr.DataArray:
        return build_dataarray(parsed)
