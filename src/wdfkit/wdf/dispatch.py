# -*- coding: utf-8 -*-
"""Classify a :class:`~wdfkit._parsed.ParsedWDF` into a scan *kind* and
dispatch to the matching handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from .error import WDFFormatError
from .handlers.registry import HANDLERS

if TYPE_CHECKING:
    from .parsed import ParsedWDF


def classify_kind(parsed: "ParsedWDF") -> str:
    """Determine the scan *kind* from a parsed WDF header.

    Returns one of:
    ``"single"``, ``"series"``, ``"points"``, ``"line_xy"``,
    ``"raster_rowmajor"``, ``"raster_columnmajor"``,
    ``"raster_snake"``, ``"linefocus"``, ``"volume"``.
    """
    for handler in HANDLERS:
        if handler.matches(parsed):
            return handler.kind
    raise WDFFormatError("scan handler match", "any handler", "none")


def dispatch(parsed: "ParsedWDF") -> xr.DataArray:
    """Classify *parsed* and call the matching handler."""
    for handler in HANDLERS:
        if handler.matches(parsed):
            return handler.build(parsed)
    raise WDFFormatError("scan handler match", "any handler", "none")
