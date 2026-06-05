# -*- coding: utf-8 -*-
"""Parse ``WXCS`` (WiRE Calibration Settings) PSET block.

Contains zero offsets, reference positions, and steps-per-unit calibration
values for each axis.
"""

from __future__ import annotations

from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext
from ..pset import parse_pset_block


def parse_wxcs(ctx: ParseContext) -> None:
    """Parse the WXCS block and store the result in ``ctx.calibration``."""
    name = "WXCS"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        body_size = ctx.blocks[i].size - 16
        ctx.f.seek(ctx.blocks[i].offset + 16)
        payload = ctx.f.read(body_size)
        ctx.calibration = parse_pset_block(payload)
