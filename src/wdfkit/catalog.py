# -*- coding: utf-8 -*-
"""Directory-level WDF catalog: fast metadata scan without loading spectra."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

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
    for p in paths:
        try:
            parsed = parse_wdf_header(p)
        except Exception:
            continue

        _SCAN_TYPES = {
            0: "Unspecified",
            1: "Static",
            2: "Continuous",
            3: "StepRepeat",
            4: "FilterScan",
            5: "FilterImage",
            6: "StreamLine",
            7: "StreamHR",
            8: "Point",
            9: "MultitrackArbitrary",
        }
        _MEAS_TYPES = {
            0: "Unspecified",
            1: "Single",
            2: "Series",
            3: "Map",
        }

        rows.append(
            {
                "filename": p.name,
                "scan_type": _SCAN_TYPES.get(
                    parsed.scan_type, str(parsed.scan_type)
                ),
                "measurement_type": _MEAS_TYPES.get(
                    parsed.measurement_type,
                    str(parsed.measurement_type),
                ),
                "nspectra": parsed.nspectra,
                "laser_wavelength": parsed.laser_wavelength,
                "laser_power": parsed.params.get("LaserPower"),
                "exposure_time": parsed.params.get("ExposureTime"),
                "xlist_units": parsed.params.get("XlistDataUnits"),
                "comment": parsed.comment,
                "start_time": parsed.acquisition_time,
                "end_time": parsed.end_time,
            }
        )

    df = pd.DataFrame(rows)
    return Catalog(df, paths)
