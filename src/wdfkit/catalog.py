# -*- coding: utf-8 -*-
"""Directory-level WDF catalog: fast metadata scan without loading spectra."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ._shared.time_utils import format_datetime
from .wdf.types import MeasurementType, ScanType, _enum_name

if TYPE_CHECKING:
    from .reader import WDFReader


class Catalog:
    """Metadata table for a collection of ``.wdf`` files.

    Build via :func:`catalog` rather than constructing directly.
    """

    def __init__(self, df: pd.DataFrame, paths: list[Path]) -> None:
        self._df = df
        self._paths = paths

    @property
    def df(self) -> pd.DataFrame:
        """The underlying metadata DataFrame (read-only view)."""
        return self._df

    def summary(self) -> pd.DataFrame:
        """Return counts and date range grouped by ``scan_type``."""
        if self._df.empty:
            return pd.DataFrame(columns=["scan_type", "count", "start", "end"])
        grp = self._df.groupby("scan_type", sort=False)
        return grp.agg(
            count=("filename", "count"),
            start=("start_time", "min"),
            end=("end_time", "max"),
        ).reset_index()

    def to_csv(self, path: str | os.PathLike[str]) -> None:
        """Write the catalog DataFrame to *path* as CSV."""
        self._df.to_csv(path, index=False)

    def load(self, idx: int) -> "WDFReader":
        """Return a fully-loaded :class:`~wdfkit.WDFReader` for row *idx*.

        *idx* is 1-based: ``load(1)`` returns the first file.
        """
        from .reader import WDFReader

        if not (1 <= idx <= len(self._paths)):
            raise IndexError(
                f"idx={idx} is out of range for catalog of "
                f"{len(self._paths)} file(s) (1-based)"
            )
        return WDFReader(self._paths[idx - 1])

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"Catalog({len(self._df)} files)"


def catalog(
    directory: str | os.PathLike[str],
    recursive: bool = False,
) -> Catalog:
    """Scan *directory* for ``.wdf`` files and return a :class:`Catalog`.

    Uses header-only parsing (no spectra loaded) — fast even for large
    collections.

    Parameters
    ----------
    directory:
        Path to the directory to scan.
    recursive:
        If ``True``, walk subdirectories recursively.
    """
    from .wdf.io import parse_wdf_header

    root = Path(directory)
    pattern = "**/*.wdf" if recursive else "*.wdf"
    paths = sorted(root.glob(pattern))

    rows = []
    good_paths: list[Path] = []
    for p in paths:
        try:
            parsed = parse_wdf_header(p)
        except Exception:
            continue

        good_paths.append(p)
        rows.append(
            {
                "filename": p.name,
                "scan_type": _enum_name(ScanType, parsed.scan_type),
                "measurement_type": _enum_name(
                    MeasurementType, parsed.measurement_type
                ),
                "nspectra": parsed.nspectra,
                "laser_wavelength": parsed.laser_wavelength,
                "laser_power": parsed.params.get("LaserPower"),
                "exposure_time": parsed.params.get("ExposureTime"),
                "xlist_units": parsed.params.get("XlistDataUnits"),
                "comment": parsed.comment,
                "start_time": format_datetime(parsed.acquisition_time),
                "end_time": format_datetime(parsed.end_time),
            }
        )

    df = pd.DataFrame(rows)
    return Catalog(df, good_paths)
