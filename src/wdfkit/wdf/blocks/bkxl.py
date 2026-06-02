# -*- coding: utf-8 -*-
"""Parse ``BKXL`` (Background X List) block.

Identical layout to XLST: 4-byte data_type, 4-byte units, then float32
values.  Present when a background reference spectrum was recorded
alongside the measurement.
"""

from __future__ import annotations

from .._helpers import constants as const
from .._helpers.binary_io import read_from_file
from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext


def parse_bkxl(ctx: ParseContext) -> None:
    """Parse the BKXL block and populate ``ctx.bkxl_*`` fields."""
    name = "BKXL"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        dt = read_from_file(ctx.f)
        ctx.bkxl_data_type = const.DATA_TYPES.get(dt, f"{dt}_unknown")
        du = read_from_file(ctx.f)
        ctx.bkxl_units = const.DATA_UNITS.get(du, f"{du}_unknown")
        n_values = int((ctx.blocks["BlockSizes"][i] - 24) / 4)
        if n_values > 0:
            ctx.bkxl_values = read_from_file(ctx.f, "<f", count=n_values)
