# -*- coding: utf-8 -*-
"""Spectral-axis naming from WiRE ``XLST`` unit enums."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SpectralAxisSpec:
    """Resolved spectral coordinate used as xarray dimension name +
    coord attrs."""

    dim_name: str
    units: str


_WIRE_UNIT_DEFAULTS: dict[str, tuple[str, str]] = {
    "Arbitrary": ("spectral", "1"),
    "RamanShift": ("raman_shift", "1/cm"),
    "Wavenumber": ("wavenumber", "cm^-1"),
    "Nanometre": ("nm", "nm"),
    "ElectronVolt": ("eV", "eV"),
    "Micron": ("micron", "µm"),
    "Counts": ("spectral_channel", "counts"),
}


def resolve_spectral_axis(
    xlist_data_units: str,
    spectral_dim: Optional[str],
) -> SpectralAxisSpec:
    """Choose spectral coordinate dimension name and
    ``coord.attrs[\"units\"]``.

    Parameters
    ----------
    xlist_data_units
        ``DATA_UNITS`` label resolved from raw XLST (e.g. ``\"Nanometre\"``).
    spectral_dim
        ``None`` or ``\"auto\"`` — derive dim name from ``xlist_data_units``.
        Any other string — force this dimension name (units still come from the
        table when known; unknown Wire enums fall back to
        ``units=\"unknown\"``).

    Returns
    -------
    SpectralAxisSpec
        ``dim_name`` is safe as an xarray dimension identifier (ASCII tokens).
    """
    fallback_units = "unknown"
    row = _WIRE_UNIT_DEFAULTS.get(xlist_data_units)
    if row:
        auto_dim, auto_units = row
    else:
        auto_dim, auto_units = ("spectral", fallback_units)

    if spectral_dim is None or spectral_dim == "auto":
        return SpectralAxisSpec(dim_name=auto_dim, units=auto_units)

    return SpectralAxisSpec(dim_name=spectral_dim, units=auto_units)
