# -*- coding: utf-8 -*-
"""Explicit ordered list of scan handlers.

Priority is visible here and nowhere else.  The first handler whose
``matches()`` returns ``True`` wins.  Ordering rules:

* Map-type flag checks come before the volume catch-all (nsteps[2] > 1),
  mirroring the original if/elif chain in ``classify_kind()``.
* ``RasterRowMajorHandler`` is the Map-level catch-all and must be placed
  after ``VolumeHandler``.
* ``SingleHandler`` is the absolute last resort.
"""

from __future__ import annotations

from .base import ScanHandler
from .line_xy import LineXYHandler
from .linefocus import LineFocusHandler
from .points import PointsHandler
from .raster_columnmajor import RasterColumnMajorHandler
from .raster_rowmajor import RasterRowMajorHandler
from .raster_snake import RasterSnakeHandler
from .series import SeriesHandler
from .single import SingleHandler
from .volume import VolumeHandler

HANDLERS: list[ScanHandler] = [
    LineXYHandler(),  # flag == XYLine
    PointsHandler(),  # flag == RandomPoints
    LineFocusHandler(),  # flag & LineFocus
    RasterSnakeHandler(),  # flag & Alternating  (before row-major)
    RasterColumnMajorHandler(),  # flag == ColumnMajor, nsteps[2] == 1
    VolumeHandler(),  # nsteps[2] > 1  (before Map catch-all)
    RasterRowMajorHandler(),  # Map catch-all (StandardRaster or unknown)
    SeriesHandler(),  # MeasurementType == Series
    SingleHandler(),  # absolute last resort
]
