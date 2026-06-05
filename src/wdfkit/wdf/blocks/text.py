# -*- coding: utf-8 -*-
"""Parse ``TEXT`` free-text comment block."""

from __future__ import annotations

from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext


def parse_text(ctx: ParseContext) -> None:
    """Read the TEXT block and store the comment in ``ctx.comment``."""
    name = "TEXT"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        body_size = ctx.blocks[i].size - 16
        ctx.f.seek(ctx.blocks[i].offset + 16)
        raw = ctx.f.read(body_size)
        nul = raw.find(b"\x00")
        text = raw[:nul] if nul >= 0 else raw
        ctx.comment = text.decode("utf-8", errors="replace")
