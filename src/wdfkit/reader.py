# -*- coding: utf-8 -*-
"""Public :class:`WDFReader` API plus module-level :func:`read` and
:func:`classify`."""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Union

import numpy as np
import xarray as xr

from .wdf.dispatch import classify_kind, dispatch
from .wdf.io import parse_wdf_header, parse_wdf_to_parsed

if TYPE_CHECKING:
    from .wdf.parsed import (
        BKXLInfo,
        OrgnEntry,
        ParsedWDF,
        WMAPInfo,
        XLSTInfo,
        YLSTInfo,
    )
    from .wdf.pset import PSet

StrPath = Union[str, os.PathLike[str]]


class WDFReader:
    """Load and expose all parsed data from a Renishaw WiRE ``.wdf`` file.

    Typical usage::

        data_array, white_light_image = WDFReader(path)

    After construction every block is accessible as a typed property.
    The xarray DataArray (shaped by scan type) is in ``.data``; the PIL
    white-light image (if any) is in ``.image``.  Both are also yielded
    by unpacking the reader directly.

    Parameters
    ----------
    spectral_dim
        Override for the spectral-axis dimension name.  ``None`` (default)
        auto-selects from the XLST units (e.g. ``RamanShift`` →
        ``"raman_shift"``).
    chunks
        Dask lazy reading: ``False`` (default, eager NumPy), ``True``
        (auto-chunk ~128 MB), or ``int`` (target MB per chunk).
    """

    def __init__(
        self,
        path: StrPath,
        *,
        verbose: bool = False,
        time_coord: str | None = "seconds_elapsed",
        spectral_dim: str | None = None,
        chunks: bool | int = False,
    ) -> None:
        self._path = os.fspath(path)
        self._verbose = verbose
        self._time_coord = time_coord
        self._spectral_dim = spectral_dim
        self._chunks = chunks

        self._parsed: ParsedWDF = parse_wdf_to_parsed(
            self._path,
            verbose=self._verbose,
            time_coord=self._time_coord,
            spectral_dim=self._spectral_dim,
            chunks=self._chunks,
        )
        if spectral_dim and spectral_dim != "auto":
            self._parsed.xlst.dim_name = spectral_dim

        self.data: xr.DataArray = dispatch(self._parsed)
        self.image = self._parsed.img

    def __iter__(self):
        yield self.data
        yield self.image

    # ------------------------------------------------------------------
    # WDF1 header scalars
    # ------------------------------------------------------------------

    @property
    def measurement_type(self) -> int:
        """Raw measurement type integer (1=Single, 2=Series, 3=Map)."""
        return self._parsed.measurement_type

    @property
    def scan_type(self) -> int:
        """Raw scan type integer."""
        return self._parsed.scan_type

    @property
    def nspectra(self) -> int:
        """Planned number of spectra (capacity)."""
        return self._parsed.nspectra

    @property
    def ncollected(self) -> int:
        """Actually collected number of spectra."""
        return self._parsed.ncollected

    @property
    def xlist_length(self) -> int:
        """Number of spectral channels per spectrum."""
        return self._parsed.npoints

    @property
    def ylist_length(self) -> int:
        """Detector Y-axis length (1 for point detectors)."""
        return self._parsed.ylist_length

    @property
    def naccum(self) -> int:
        """Accumulations per spectrum."""
        return self._parsed.naccum

    @property
    def app_name(self) -> str:
        """WiRE application name string."""
        return self._parsed.app_name

    @property
    def app_version(self) -> str:
        """WiRE application version string."""
        return self._parsed.app_version

    @property
    def file_uuid(self) -> str:
        """Unique file identifier from WDF1 header."""
        return self._parsed.file_uuid

    # ------------------------------------------------------------------
    # Spectral & secondary axes (§4, §5)
    # ------------------------------------------------------------------

    @property
    def xlst(self) -> XLSTInfo:
        """XLST block: spectral axis values, data_type, units, dim_name."""
        return self._parsed.xlst

    @property
    def ylst(self) -> YLSTInfo | None:
        """YLST block: detector-Y axis (None for point detectors)."""
        return self._parsed.ylst

    # ------------------------------------------------------------------
    # Raw spectral data (§3)
    # ------------------------------------------------------------------

    @property
    def raw_data(self) -> np.ndarray | None:
        """Flat spectral array of shape (nspectra, xlist_length), float32."""
        return self._parsed.data

    # ------------------------------------------------------------------
    # Per-spectrum origins (§6)
    # ------------------------------------------------------------------

    @property
    def orgn(self) -> list[OrgnEntry]:
        """List of ORGN entries (spatial / time / flags per spectrum)."""
        return self._parsed.orgn

    def orgn_by_type(self, data_type: str) -> OrgnEntry | None:
        """Return the first ORGN entry matching *data_type*."""
        return self._parsed.orgn_by_type(data_type)

    # ------------------------------------------------------------------
    # Text comment (§7)
    # ------------------------------------------------------------------

    @property
    def comment(self) -> str | None:
        """Free-text comment from the TEXT block."""
        return self._parsed.comment

    # ------------------------------------------------------------------
    # Map geometry (§8)
    # ------------------------------------------------------------------

    @property
    def wmap(self) -> WMAPInfo | None:
        """WMAP block: grid geometry. ``None`` if not a map scan."""
        return self._parsed.wmap

    # ------------------------------------------------------------------
    # White-light image (§9)
    # ------------------------------------------------------------------

    @property
    def whtl_jpeg_bytes(self) -> bytes | None:
        """Raw JPEG bytes from the WHTL block, or ``None`` if absent."""
        return self._parsed.whtl_jpeg_bytes

    @property
    def has_whitelight(self) -> bool:
        """``True`` if a WHTL white-light image block is present."""
        return self._parsed.whtl_jpeg_bytes is not None

    # ------------------------------------------------------------------
    # PSET blocks (§10)
    # ------------------------------------------------------------------

    @property
    def acquisition(self) -> PSet | None:
        """WXDA block parsed as a PSet (scan / acquisition properties)."""
        return self._parsed.acquisition

    @property
    def instrument_status(self) -> PSet | None:
        """WXIS block parsed as a PSet (motor positions, instrument state)."""
        return self._parsed.instrument_status

    @property
    def calibration(self) -> PSet | None:
        """WXCS block parsed as a PSet (calibration settings)."""
        return self._parsed.calibration

    @property
    def zeldac(self) -> PSet | None:
        """ZLDC block parsed as a PSet (zero level & dark current)."""
        return self._parsed.zeldac

    # ------------------------------------------------------------------
    # Derived convenience accessors
    # ------------------------------------------------------------------

    @property
    def initial_coordinates(self) -> dict | None:
        """Stage XYZ at acquisition time from WXIS.

        Returns ``{"x_um", "y_um", "z_um", "x_str", "y_str", "z_str"}``
        for every measurement type, including Single scans where ORGN
        carries no spatial origins.
        """
        return self._parsed.initial_coordinates

    @property
    def motor_positions(self) -> dict | None:
        """All motor positions from WXIS as ``{label: (µm, string)}``."""
        return self._parsed.motor_positions

    @property
    def acquisition_time(self) -> datetime | None:
        """Acquisition start time decoded from the ORGN Time entry."""
        return self._parsed.acquisition_time

    # ------------------------------------------------------------------
    # Background X list (§11)
    # ------------------------------------------------------------------

    @property
    def bkxl(self) -> BKXLInfo | None:
        """BKXL block: background X list (mirrored spectral axis)."""
        return self._parsed.bkxl


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------


def read(
    path: StrPath,
    *,
    verbose: bool = False,
    time_coord: str | None = "seconds_elapsed",
    spectral_dim: str | None = None,
    chunks: bool | int = False,
) -> xr.DataArray:
    """Read a WiRE ``.wdf`` file and return a :class:`xarray.DataArray`.

    Parameters
    ----------
    path
        Path to the ``.wdf`` file.
    time_coord
        Name for the elapsed-time coordinate on Series scans.
        ``None`` disables it (absolute timestamps used instead).
    spectral_dim
        Override for the spectral-axis dimension name.
    chunks
        Dask chunking: ``False`` (eager), ``True`` (auto), or int (target MB).

    Returns
    -------
    xarray.DataArray
        Shape and dims depend on scan kind; spectral axis is always last.
    """
    parsed = parse_wdf_to_parsed(
        path,
        verbose=verbose,
        time_coord=time_coord,
        spectral_dim=spectral_dim,
        chunks=chunks,
    )
    if spectral_dim and spectral_dim != "auto":
        parsed.xlst.dim_name = spectral_dim
    return dispatch(parsed)


def classify(path: StrPath) -> dict:
    """Return scan classification for a WiRE ``.wdf`` file *without*
    loading the spectral data.

    Returns
    -------
    dict
        Keys: ``kind``, ``measurement_type``, ``scan_type``,
        ``wmap_flag``, ``nspectra``, ``npoints``, ``nsteps``.
    """
    parsed = parse_wdf_header(path)
    kind = classify_kind(parsed)
    info: dict = {
        "kind": kind,
        "measurement_type": parsed.params.get("MeasurementType", ""),
        "scan_type": parsed.params.get("ScanType", ""),
        "wmap_flag": parsed.wmap.flag if parsed.wmap else None,
        "nspectra": parsed.nspectra,
        "npoints": parsed.npoints,
        "nsteps": (
            parsed.wmap.nsteps.tolist() if parsed.wmap is not None else None
        ),
    }
    return info
