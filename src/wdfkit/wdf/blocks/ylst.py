# -*- coding: utf-8 -*-
"""Parse ``YLST`` block (secondary Y-axis list when present)."""

from __future__ import annotations

from .._helpers import constants as const
from .._helpers.binary_io import read_from_file
from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext


def parse_ylst(ctx: ParseContext) -> None:
    name = "YLST"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        yldt = read_from_file(ctx.f)
        ctx.params["YlistDataType"] = const.DATA_TYPES.get(
            yldt, f"{yldt}_unknown"
        )
        yldu = read_from_file(ctx.f)
        ctx.params["YlistDataUnits"] = const.DATA_UNITS.get(
            yldu, f"{yldu}_unknown"
        )
        y_values_count = int((ctx.blocks["BlockSizes"][i] - 24) / 4)
        if y_values_count > 1:
            y_values = read_from_file(ctx.f, "<f", count=y_values_count)
            ctx.ylst_values = y_values
            ctx.ylst_data_type = ctx.params.get("YlistDataType", "")
            ctx.ylst_units = ctx.params.get("YlistDataUnits", "")
            if ctx.verbose:
                print(y_values)
