# -*- coding: utf-8 -*-
"""Parse ``ZLDC`` (Zero Level & Dark Current) PSET block.

Small block (~300 B) containing ZeldacType, ZeroLevelAndDarkCurrent,
and an user identifier.
"""

from __future__ import annotations

from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext
from ..pset import parse_pset_block


def parse_zldc(ctx: ParseContext) -> None:
    """Parse the ZLDC block and store the result in ``ctx.zeldac``."""
    name = "ZLDC"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        body_size = ctx.blocks["BlockSizes"][i] - 16
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        payload = ctx.f.read(body_size)
        ctx.zeldac = parse_pset_block(payload)
