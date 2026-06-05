# -*- coding: utf-8 -*-
"""Shared ``make_attrs`` helper used by all handler modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from ..._shared.time_utils import format_datetime
from .._helpers.utils import hr_filesize

if TYPE_CHECKING:
    from ..parsed import ParsedWDF


def spectral_coord(parsed: "ParsedWDF") -> tuple:
    """Return ``("spectral", values, attrs)`` for the spectral coord."""
    attrs: dict = {}
    if parsed.xlst.coord_units:
        attrs["units"] = parsed.xlst.coord_units
    attrs["long_name"] = parsed.xlst.units
    return "spectral", parsed.xlst.values, attrs


def sort_spectral(da: "xr.DataArray") -> "xr.DataArray":
    """Sort *da* along its last (spectral) dimension in ascending order.

    This mirrors the ``da.sortby(sdim)`` call that the legacy assembler
    applied so that spectral values always run low→high.
    """
    sdim = da.dims[-1]
    coord = da[sdim].values
    if len(coord) > 1 and coord[0] > coord[-1]:
        da = da.isel({sdim: slice(None, None, -1)})
    return da


def make_attrs(parsed: "ParsedWDF", kind: str) -> dict:
    """Build the user-facing attrs dict for a DataArray.

    Only scientifically relevant keys are included.  Internal parser fields
    (WdfFlag, Capacity, ApplicationName, etc.) are intentionally omitted.
    Map-geometry keys (MapAreaType, StepSizes, NSteps, LineFocusSize) are
    present only when a WMAP block exists.
    """
    p = parsed.params

    attrs: dict = {
        "title": p.get("Title", ""),
        "comment": parsed.comment,
        "laser_wavelength_nm": p.get("LaserWaveLength"),
        "scan_type": p.get("ScanType"),
        "measurement_type": p.get("MeasurementType"),
        "n_spectra": p.get("Count"),
        "n_points": p.get("PointsPerSpectrum"),
        "n_accumulations": p.get("AccumulationCount"),
        "start_time": format_datetime(parsed.acquisition_time),
        "end_time": format_datetime(parsed.end_time),
        "spectral_units": parsed.xlst.units,
        "file_size": hr_filesize(parsed.filesize),
        "kind": kind,
        "treatments": {},
    }

    # Optional acquisition settings (present only when WXDA block exists)
    if parsed.exposure_time is not None:
        attrs["exposure_time"] = parsed.exposure_time
    if parsed.laser_power is not None:
        attrs["laser_power"] = parsed.laser_power

    # Stage / map coordinates
    if parsed.wmap is not None:
        mp = parsed.map_params
        ic = mp.get("InitialCoordinates")
        attrs["initial_coordinates"] = (
            {"x": float(ic[0]), "y": float(ic[1]), "z": float(ic[2])}
            if ic is not None and not isinstance(ic, dict)
            else ic
        )
        attrs["map_type"] = mp.get("MapAreaType")
        attrs["step_sizes_um"] = mp.get("StepSizes")
        attrs["n_steps"] = mp.get("NbSteps")
        attrs["line_focus_size"] = mp.get("LineFocusSize")
    else:
        attrs["initial_coordinates"] = parsed.stage_xyz

    return attrs
