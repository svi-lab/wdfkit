# -*- coding: utf-8 -*-
"""``ParsedWDF`` — immutable result of WiRE block parsing, consumed by
handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np


@dataclass
class XLSTInfo:
    """Decoded XLST (spectral-axis) block."""

    values: np.ndarray
    data_type: str  # e.g. "Spectral"
    units: str  # e.g. "RamanShift"
    dim_name: str  # resolved xarray dim name, e.g. "raman_shift"
    coord_units: str = (
        ""  # human-readable units for xarray coord attrs, e.g. "nm"
    )


@dataclass
class YLSTInfo:
    """Decoded YLST block (line-focus secondary axis)."""

    values: np.ndarray
    data_type: str
    units: str


@dataclass
class BKXLInfo:
    """Decoded BKXL (background X list) block."""

    values: np.ndarray
    data_type: str
    units: str


@dataclass
class OrgnEntry:
    """Single origin-set entry from the ORGN block."""

    label: str  # as stored in file, e.g. "X", "Y", "Z", "Time"
    data_type: str  # DATA_TYPES string, e.g. "SpatialX"
    units: str  # DATA_UNITS string (lowercased), e.g. "micron"
    values: np.ndarray
    is_primary: bool  # high bit (0x80000000) was set in the raw type field


@dataclass
class WMAPInfo:
    """Decoded WMAP block."""

    flag: int  # raw bitmask integer
    start_xyz: np.ndarray  # 3-element float32 array
    step_xyz: np.ndarray  # 3-element float32 array
    nsteps: np.ndarray  # 3-element uint32 array [nx, ny, nz]
    linefocus_size: int


@dataclass
class ParsedWDF:
    """Fully-decoded WDF file, ready for handler consumption.

    Handlers receive a ``ParsedWDF`` and must NOT re-open the file.
    """

    filename: str
    filesize: int

    # ---- scalars from WDF1 ----
    nspectra: int  # Capacity (planned)
    ncollected: int  # Count (actual)
    npoints: int  # PointsPerSpectrum / xlist_length
    ylist_length: int
    measurement_type: int  # raw int: 1=Single, 2=Series, 3=Map
    scan_type: int  # raw int
    app_name: str
    app_version: str
    naccum: int
    laser_wavelength: Any  # float or "Unspecified"
    title: str
    spectral_units_str: str  # e.g. "Counts"

    # ---- full params dicts (for attrs passthrough) ----
    params: dict  # WDF1 decoded params
    map_params: dict  # WMAP decoded params (empty if no WMAP)

    # ---- block data ----
    data: Optional[
        np.ndarray
    ]  # shape (nspectra, npoints); None for header-only
    xlst: XLSTInfo
    ylst: Optional[YLSTInfo]
    orgn: list[OrgnEntry] = field(default_factory=list)
    wmap: Optional[WMAPInfo] = None

    # ---- extras ----
    img: Any = None  # PIL image or None
    exposure_time: Optional[float] = None  # seconds
    laser_power: Optional[float] = None  # percent
    stage_xyz: Optional[dict] = None  # {"x": µm, "y": µm, "z": µm} from WXIS

    # ---- new block data (§7–§11) ----
    comment: Optional[str] = None  # TEXT block free text
    acquisition: Any = None  # WXDA PSet
    instrument_status: Any = None  # WXIS PSet
    calibration: Any = None  # WXCS PSet
    zeldac: Any = None  # ZLDC PSet
    bkxl: Any = None  # BKXL XList (values, data_type, units)
    whtl_jpeg_bytes: Optional[bytes] = None  # raw JPEG from WHTL block
    initial_coordinates: Optional[dict] = (
        None  # {"x_um","y_um","z_um","x_str","y_str","z_str"}
    )
    motor_positions: Optional[dict] = None
    acquisition_time: Optional[datetime] = None  # decoded from ORGN Time
    end_time: Optional[datetime] = None  # from WDF1 header
    file_uuid: str = ""  # from WDF1 header

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def orgn_by_type(self, data_type: str) -> Optional[OrgnEntry]:
        """Return the first ORGN entry whose ``data_type`` matches."""
        return next((e for e in self.orgn if e.data_type == data_type), None)

    def primary_orgn(self) -> Optional[OrgnEntry]:
        """Return the first ORGN entry with the primary-axis flag set."""
        return next((e for e in self.orgn if e.is_primary), None)
