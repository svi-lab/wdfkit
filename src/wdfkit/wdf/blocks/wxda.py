# -*- coding: utf-8 -*-
"""Parse ``WXDA`` (WiRE Data Acquisition) PSET block.

Contains scan / acquisition properties: CCD serial number, exposure times,
PAF settings, bleach times, etc.
"""

from __future__ import annotations

from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext
from ..pset import parse_pset_block


def parse_wxda(ctx: ParseContext) -> None:
    """Parse the WXDA block and store the result in ``ctx.acquisition``."""
    name = "WXDA"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        body_size = ctx.blocks["BlockSizes"][i] - 16
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        payload = ctx.f.read(body_size)
        ctx.acquisition = parse_pset_block(payload)
