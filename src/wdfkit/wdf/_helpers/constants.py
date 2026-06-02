#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renishaw WiRE .wdf enumeration tables (ported from legacy
constants_WDF_class)."""

import numpy as np


def list_to_dict(list_name):
    return dict(zip(range(len(list_name)), list_name))


MEASUREMENT_TYPES_LIST = [
    "Unspecified",
    "Single",
    "Series",
    "Map",
]
MEASUREMENT_TYPES = list_to_dict(MEASUREMENT_TYPES_LIST)

SCAN_TYPES_LIST = [
    "Unspecified",
    "Static",
    "Continuous",
    "StepRepeat",
    "FilterScan",
    "FilterImage",
    "StreamLine",
    "StreamLineHR",
    "Point",
    "MultitrackDiscrete",
    "LineFocusMapping",
]
SCAN_TYPES = list_to_dict(SCAN_TYPES_LIST)

MAP_TYPES = {
    0: "RandomPoints",  # 1 << 0
    1: "ColumnMajor",  # 1 << 1
    2: "Alternating2",  #
    3: "LineFocusMapping",  # 1 << 3
    4: "InvertedRows",  # 1 << 4 (deprecated)
    5: "InvertedColumns",  # 1 << 5 (deprecated)
    6: "SurfaceProfile",  # 1 << 6
    7: "XyLine",  # 1 << 7
    64: "LiveTrack",
    66: "StreamLine",
    68: "InvertedRows2",  # Remember to check this 68
    128: "Slice",
    192: "Corrine",
}

DATA_TYPES_LIST = [
    "Arbitrary",
    "Spectral",
    "Intensity",
    "SpatialX",
    "SpatialY",
    "SpatialZ",
    "SpatialR",
    "SpatialTheta",
    "SpatialPhi",
    "Temperature",
    "Pressure",
    "Time",
    "Derived",
    "Polarization",
    "FocusTrack",
    "RampRate",
    "Checksum",
    "Flags",
    "ElapsedTime",
    "Frequency",
    "MpWellSpatialX",
    "MpWellSpatialY",
    "MpLocationIndex",
    "MpWellReference",
    "PAFZActual",
    "PAFZError",
    "PAFSignalUsed",
    "ExposureTime",
    "EndMarker",
]
DATA_TYPES = list_to_dict(DATA_TYPES_LIST)
DATA_TYPES[1024] = "1024_Fatima_BigFile"


DATA_UNITS_LIST = [
    "Arbitrary",
    "RamanShift",
    "Wavenumber",
    "Nanometre",
    "ElectronVolt",
    "Micron",
    "Counts",
    "Electrons",
    "Millimetres",
    "Metres",
    "Kelvin",
    "Pascal",
    "Seconds",
    "Milliseconds",
    "Hours",
    "Days",
    "Pixels",
    "Intensity",
    "RelativeIntensity",
    "Degrees",
    "Radians",
    "Celsius",
    "Fahrenheit",
    "KelvinPerMinute",
    "FileTime",
    "Microseconds",
    "EndMarker",
]
DATA_UNITS = list_to_dict(DATA_UNITS_LIST)

WDF_FLAGS = {
    0: "WdfXYXY",
    1: "WdfChecksum",
    2: "WdfCosmicRayRemoval",
    3: "WdfMultitrack",
    4: "WdfSaturation",
    5: "WdfFileBackup",
    6: "WdfTemporary",
    7: "WdfSlice",
    8: "WdfPQ",
    16: "16: UnknownFlag (LiveTrack?)",
}

EXIF_TAGS = {
    # Renishaw's particular tags:
    65184: "FocalPlaneXYOrigins",  # tuple of floats
    65185: "FieldOfViewXY",  # tuple of floats
    65186: "px/µ ?",  # float (1.0, 5.0?)
    # Standard Exif tags:
    34665: "ExifOffset",  # Normally, (114?)
    270: "ImageDescription",  # Normally, "white-light image"
    271: "Make",  # Normally, "Renishaw"
    41488: "FocalPlaneResolutionUnit",  # (`5` corresponds to microns)
    41486: "FocalPlaneXResolution",  # (27120.6?)
    41487: "FocalPlaneYResolution",  # (21632.1?)
}

HEADER_DT = np.dtype(
    [
        ("block_name", "|S4"),
        ("block_id", np.int32),
        ("block_size", np.int64),
    ]
)
