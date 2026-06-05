# -*- coding: utf-8 -*-
"""Parse ``WXIS`` (WiRE Instrument State) block.

The ``WXIS`` block stores a snapshot of all hardware motor positions and
instrument configuration at the time of acquisition.

Exposed on context
------------------
``ctx.instrument_status``
    Full :class:`~wdfkit.wdf.pset.PSet` for the block.
``ctx.initial_coordinates``
    ``{"x_um", "y_um", "z_um", "x_str", "y_str", "z_str"}`` stage XYZ.
``ctx.motor_positions``
    ``{motor_label: (numeric_um, string_value)}`` for every motor entry.
``ctx.stage_xyz``
    ``{"x": µm, "y": µm, "z": µm}`` kept for backward compatibility.
``ctx.params["LaserPower"]``
    ND-filter transmission in percent when present.
"""

from __future__ import annotations

from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext
from ..pset import parse_pset_block

_XYZ_LABELS = (
    "XYZ Stage X Motor",
    "XYZ Stage Y Motor",
    "XYZ Stage Z Motor",
)


def _parse_nd_percent(raw: object) -> float | None:
    """Convert a raw ``"ND Transmission %"`` value to a plain float."""
    if raw is None:
        return None
    s = str(raw).strip()
    try:
        return float(s)
    except ValueError:
        pass
    s_lower = s.lower()
    for suffix in (" percent", "%"):
        if s_lower.endswith(suffix):
            try:
                return float(s[: len(s) - len(suffix)].strip())
            except ValueError:
                pass
    return None


def parse_wxis(ctx: ParseContext) -> None:
    """Parse the WXIS block.

    Updates ``ctx.instrument_status``, ``ctx.initial_coordinates``,
    ``ctx.motor_positions``, ``ctx.stage_xyz``, and
    ``ctx.params["LaserPower"]``.  Missing or malformed blocks are silently
    ignored.
    """
    name = "WXIS"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        block = ctx.blocks[i]
        ctx.f.seek(block.offset + 16)
        payload = ctx.f.read(block.size - 16)

        pset = parse_pset_block(payload)
        ctx.instrument_status = pset

        # ---- Laser power ----
        nd = _parse_nd_percent(pset.get_by_label("ND Transmission %"))
        if nd is not None:
            ctx.params["LaserPower"] = nd

        # ---- Stage XYZ (numeric µm from the PSet) ----
        x_label, y_label, z_label = _XYZ_LABELS
        x_val = pset.get_by_label(x_label)
        y_val = pset.get_by_label(y_label)
        z_val = pset.get_by_label(z_label)

        def _to_um(v: object) -> float | None:
            return float(v) if isinstance(v, (int, float)) else None

        x_um = _to_um(x_val)
        y_um = _to_um(y_val)
        z_um = _to_um(z_val)

        if any(v is not None for v in (x_um, y_um, z_um)):
            ctx.stage_xyz = {
                "x": float(x_um if x_um is not None else 0.0),
                "y": float(y_um if y_um is not None else 0.0),
                "z": float(z_um if z_um is not None else 0.0),
            }
            ctx.initial_coordinates = {
                "x_um": float(x_um if x_um is not None else 0.0),
                "y_um": float(y_um if y_um is not None else 0.0),
                "z_um": float(z_um if z_um is not None else 0.0),
                "x_str": str(x_val) if x_val is not None else "",
                "y_str": str(y_val) if y_val is not None else "",
                "z_str": str(z_val) if z_val is not None else "",
            }
