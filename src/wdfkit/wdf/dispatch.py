# -*- coding: utf-8 -*-
"""Classify a :class:`~wdfkit._parsed.ParsedWDF` into a scan *kind* and
dispatch to the matching handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import xarray as xr

from .handlers import (
    line_xy,
    linefocus,
    points,
    raster_columnmajor,
    raster_rowmajor,
    raster_snake,
    series,
    single,
    volume,
)
from .types import MapFlag, MeasurementType

if TYPE_CHECKING:
    from .parsed import ParsedWDF

# Dispatch table: kind string → handler function
_HANDLERS: dict[str, Callable[["ParsedWDF"], xr.DataArray]] = {
    "single": single.build_dataarray,
    "series": series.build_dataarray,
    "points": points.build_dataarray,
    "line_xy": line_xy.build_dataarray,
    "raster_rowmajor": raster_rowmajor.build_dataarray,
    "raster_columnmajor": raster_columnmajor.build_dataarray,
    "raster_snake": raster_snake.build_dataarray,
    "linefocus": linefocus.build_dataarray,
    "volume": volume.build_dataarray,
}


def classify_kind(parsed: "ParsedWDF") -> str:
    """Determine the scan *kind* from a parsed WDF header.

    Returns one of:
    ``"single"``, ``"series"``, ``"points"``, ``"line_xy"``,
    ``"raster_rowmajor"``, ``"raster_columnmajor"``,
    ``"raster_snake"``, ``"linefocus"``, ``"volume"``.
    """
    mt = parsed.measurement_type
    if mt == MeasurementType.Single:
        return "single"
    if mt == MeasurementType.Series:
        return "series"
    if mt == MeasurementType.Map:
        assert parsed.wmap is not None, "Map scan missing WMAP block"
        flag = parsed.wmap.flag
        nsteps = parsed.wmap.nsteps
        if flag == MapFlag.XYLine:
            return "line_xy"
        if flag == MapFlag.RandomPoints:
            return "points"
        if flag & MapFlag.LineFocus:
            return "linefocus"
        if flag & MapFlag.Alternating:
            return "raster_snake"
        if flag == MapFlag.ColumnMajor and nsteps[2] == 1:
            return "raster_columnmajor"
        if flag == MapFlag.StandardRaster and nsteps[2] == 1:
            return "raster_rowmajor"
        if nsteps[2] > 1:
            return "volume"
        # Unknown combination — treat as raster_rowmajor
        return "raster_rowmajor"
    # Unspecified or unknown MeasurementType
    return "single"


def dispatch(parsed: "ParsedWDF") -> xr.DataArray:
    """Classify *parsed* and call the matching handler."""
    kind = classify_kind(parsed)
    handler = _HANDLERS.get(kind)
    if handler is None:
        raise ValueError(f"No handler registered for kind={kind!r}")
    return handler(parsed)
