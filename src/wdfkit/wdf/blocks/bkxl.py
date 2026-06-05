# -*- coding: utf-8 -*-
"""Parse ``BKXL`` (Background X List) block.

Identical layout to XLST: 4-byte data_type, 4-byte units, then float32
values.  Present when a background reference spectrum was recorded
alongside the measurement.
"""

from __future__ import annotations

from .._helpers.binary_io import read_from_file
from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext
from ..types import DataType, UnitType, _enum_name


def parse_bkxl(ctx: ParseContext) -> None:
    """Parse the BKXL block and populate ``ctx.bkxl_*`` fields."""
    name = "BKXL"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks[i].offset + 16)
        dt = read_from_file(ctx.f)
        ctx.bkxl_data_type = _enum_name(DataType, dt)
        du = read_from_file(ctx.f)
        ctx.bkxl_units = _enum_name(UnitType, du)
        n_values = int((ctx.blocks[i].size - 24) / 4)
        if n_values > 0:
            ctx.bkxl_values = read_from_file(ctx.f, "<f", count=n_values)
