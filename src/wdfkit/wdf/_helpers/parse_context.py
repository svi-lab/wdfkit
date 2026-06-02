# -*- coding: utf-8 -*-
"""Mutable parse state passed through WiRE block parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, Any, Optional


@dataclass
class ParseContext:
    """Shared state while reading a single ``.wdf`` file."""

    filename: str
    verbose: bool
    time_coord: Optional[str]
    spectral_dim: Optional[str]
    filesize: int = 0
    f: Optional[IO[bytes]] = None
    params: dict = field(default_factory=dict)
    map_params: dict = field(default_factory=dict)
    coord_dict: dict = field(default_factory=dict)
    blocks: dict = field(default_factory=dict)
    spectra: Any = None  # np.ndarray or dask.array set in DATA block
    npoints: int = 0
    nspectra: int = 0
    ncollected: int = 0
    spectral_dim_name: Optional[str] = None
    img: Any = None
    origin_labels: list = field(default_factory=list)
    origin_set_dtypes: list = field(default_factory=list)
    origin_set_units: list = field(default_factory=list)
    origin_is_primary: list = field(default_factory=list)
    chunks: bool | int = False  # False = eager; True or int MB = lazy/dask
    # YLST values (stored for linefocus handler)
    ylst_values: object = None  # np.ndarray or None
    ylst_data_type: str = ""
    ylst_units: str = ""
    # Raw integer fields needed by ParsedWDF / classify
    measurement_type_raw: int = 0
    scan_type_raw: int = 0
    wmap_flag_raw: int = 0
    # Stage position from WXIS (all measurement types)
    stage_xyz: Optional[dict] = None
    # New block data
    comment: Optional[str] = None  # TEXT block
    acquisition: Any = None  # WXDA PSet
    instrument_status: Any = None  # WXIS PSet
    calibration: Any = None  # WXCS PSet
    zeldac: Any = None  # ZLDC PSet
    bkxl_values: Any = None  # BKXL float32 array
    bkxl_data_type: str = ""
    bkxl_units: str = ""
    whtl_jpeg_bytes: Optional[bytes] = None  # raw JPEG from WHTL
    initial_coordinates: Optional[dict] = None  # from WXIS
    motor_positions: Optional[dict] = None  # from WXIS

    def print_block_header(self, name: str, index: int) -> None:
        if self.verbose:
            print(
                f"\n{' Block : ' + name + ' ':=^80s}\n"
                f"size: {self.blocks['BlockSizes'][index]},"
                f"offset: {self.blocks['BlockOffsets'][index]}"
            )
