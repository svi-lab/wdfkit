# -*- coding: utf-8 -*-
"""Enums and spectral-axis naming for WiRE .wdf scan classification."""

from __future__ import annotations

from enum import IntEnum


class MeasurementType(IntEnum):
    Unspecified = 0
    Single = 1
    Series = 2
    Map = 3


class ScanType(IntEnum):
    Unspecified = 0
    Static = 1
    Continuous = 2
    StepRepeat = 3
    FilterScan = 4
    FilterImage = 5
    StreamLine = 6
    StreamLineHR = 7
    Point = 8
    MultitrackDiscrete = 9
    LineFocusMapping = 10


class MapFlag:
    """Bitmask constants for the WMAP ``flag`` field."""

    StandardRaster = 0x00
    RandomPoints = 0x01
    ColumnMajor = 0x02
    Alternating = 0x04
    LineFocus = 0x08
    InvertedRows = 0x10
    InvertedColumns = 0x20
    SurfaceProfile = 0x40
    XYLine = 0x80


class DataType(IntEnum):
    Arbitrary = 0
    Spectral = 1
    Intensity = 2
    SpatialX = 3
    SpatialY = 4
    SpatialZ = 5
    SpatialR = 6
    SpatialTheta = 7
    SpatialPhi = 8
    Temperature = 9
    Pressure = 10
    Time = 11
    Derived = 12
    Polarization = 13
    FocusTrack = 14
    RampRate = 15
    Checksum = 16
    Flags = 17
    ElapsedTime = 18
    Frequency = 19
    MpWellSpatialX = 20
    MpWellSpatialY = 21
    MpLocationIndex = 22
    MpWellReference = 23
    PAFZActual = 24
    PAFZError = 25
    PAFSignalUsed = 26
    ExposureTime = 27
    EndMarker = 28


class UnitType(IntEnum):
    Arbitrary = 0
    RamanShift = 1
    Wavenumber = 2
    Nanometre = 3
    ElectronVolt = 4
    Micrometre = 5
    Counts = 6
    Electrons = 7
    Millimetres = 8
    Metres = 9
    Kelvin = 10
    Pascal = 11
    Seconds = 12
    Milliseconds = 13
    Hours = 14
    Days = 15
    Pixels = 16
    Intensity = 17
    RelativeIntensity = 18
    Degrees = 19
    Radians = 20
    Celsius = 21
    Fahrenheit = 22
    KelvinPerMinute = 23
    FileTime = 24
    Microseconds = 25


# Maps XLST units string (from internal/constants.py DATA_UNITS_LIST)
# to xarray dimension name for the spectral axis.
_SPECTRAL_DIM_BY_UNIT: dict[str, str] = {
    "Arbitrary": "spectral",
    "RamanShift": "raman_shift",
    "Wavenumber": "wavenumber",
    "Nanometre": "wavelength_nm",
    "ElectronVolt": "energy_eV",
    "Micron": "wavelength_um",
    "Pixels": "pixel",
    "Counts": "spectral_channel",
}

_SPECTRAL_DIM_BY_DTYPE: dict[str, str] = {
    "Spectral": "spectral",
    "Arbitrary": "spectral",
}


def get_spectral_dim_name(
    xlst_units: str,
    xlst_data_type: str = "Spectral",
    override: str | None = None,
) -> str:
    """Return the xarray dimension name for the spectral axis.

    Prefers *xlst_units* (more specific), falls back to *xlst_data_type*,
    then to ``"spectral"``.  If *override* is given (not ``None`` or
    ``"auto"``), it is returned as-is.
    """
    if override and override != "auto":
        return override
    return (
        _SPECTRAL_DIM_BY_UNIT.get(xlst_units)
        or _SPECTRAL_DIM_BY_DTYPE.get(xlst_data_type)
        or "spectral"
    )
