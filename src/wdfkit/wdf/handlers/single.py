# -*- coding: utf-8 -*-
"""Handler for ``kind="single"`` (MeasurementType == 1, nspectra == 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._attrs import make_attrs, sort_spectral, spectral_coord

if TYPE_CHECKING:
    from ..parsed import ParsedWDF


def build_dataarray(parsed: "ParsedWDF") -> xr.DataArray:
    """Return a 1-D DataArray ``(spectral_dim,)``."""
    sdim = parsed.xlst.dim_name
    data = np.asarray(parsed.data)
    if data.ndim == 2:
        spectrum = data[0]
    else:
        spectrum = data

    attrs = make_attrs(parsed, "single")
    attrs["ScanShape"] = (1, 1)
    attrs["RowCoord"] = None
    attrs["ColCoord"] = None

    sc = spectral_coord(parsed)
    da = xr.DataArray(
        spectrum,
        dims=(sdim,),
        coords={sdim: (sdim, sc[1], sc[2])},
        attrs=attrs,
    )
    return sort_spectral(da)
