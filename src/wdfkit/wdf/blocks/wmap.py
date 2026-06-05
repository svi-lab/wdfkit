# -*- coding: utf-8 -*-
"""Parse ``WMAP`` map-area metadata block."""

from __future__ import annotations

import numpy as np

from .._helpers.binary_io import read_from_file
from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext

_MAP_AREA_TYPE: dict[int, str] = {
    0x00: "StandardRaster",
    0x01: "RandomPoints",
    0x02: "ColumnMajor",
    0x04: "Alternating",
    0x08: "LineFocus",
    0x10: "InvertedRows",
    0x20: "InvertedColumns",
    0x40: "SurfaceProfile",
    0x80: "XYLine",
}


def parse_wmap(ctx: ParseContext) -> None:
    name = "WMAP"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        ctx.f.seek(ctx.blocks[i].offset + 16)
        m_flag = int(read_from_file(ctx.f))
        ctx.wmap_flag_raw = m_flag
        ctx.map_params["MapAreaType"] = _MAP_AREA_TYPE.get(m_flag, str(m_flag))
        read_from_file(ctx.f)
        ctx.map_params["InitialCoordinates"] = np.round(
            read_from_file(ctx.f, "<f", count=3), 2
        )
        ctx.map_params["StepSizes"] = np.round(
            read_from_file(ctx.f, "<f", count=3), 2
        )
        ctx.map_params["NbSteps"] = read_from_file(ctx.f, np.uint32, count=3)
        ctx.map_params["LineFocusSize"] = read_from_file(ctx.f)
