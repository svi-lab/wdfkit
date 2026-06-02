# -*- coding: utf-8 -*-
"""WiRE ``.wdf`` binary parsing — blocks, context, handlers."""

from .dispatch import classify_kind, dispatch
from .io import parse_wdf_header, parse_wdf_to_parsed, read_wdf_file
from .parsed import ParsedWDF
from .types import MapFlag, MeasurementType, ScanType

__all__ = [
    "classify_kind",
    "dispatch",
    "MapFlag",
    "MeasurementType",
    "parse_wdf_header",
    "parse_wdf_to_parsed",
    "ParsedWDF",
    "read_wdf_file",
    "ScanType",
]
