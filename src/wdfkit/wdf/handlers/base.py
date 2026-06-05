# -*- coding: utf-8 -*-
"""Abstract base class for scan handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from ..parsed import ParsedWDF


class ScanHandler(ABC):
    """Interface that every scan-kind handler must implement.

    ``matches()`` decides whether this handler owns the parsed file.
    ``build()`` assembles the final :class:`xarray.DataArray`.
    Both receive the same ``ParsedWDF`` — no double-parsing.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Canonical kind string, e.g. ``"raster_rowmajor"``."""

    @abstractmethod
    def matches(self, parsed: "ParsedWDF") -> bool:
        """Return ``True`` if this handler should handle *parsed*."""

    @abstractmethod
    def build(self, parsed: "ParsedWDF") -> xr.DataArray:
        """Assemble and return the shaped :class:`xarray.DataArray`."""
