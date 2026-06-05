# -*- coding: utf-8 -*-
"""Parse ``WXDM`` (WiRE Data Measurement) block.

The ``WXDM`` block stores the VBScript acquisition script together with a
flat PSET property table.  The property table uses inherited key definitions
(nested PSETs share the top-level key map), so the decoder recurses into
nested blobs when they carry no key definitions of their own.

Currently extracted
-------------------
``ExposureTime``
    CCD exposure time in **seconds** (stored as milliseconds in the file).
"""

from __future__ import annotations

from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext
from ..pset import decode_pset


def parse_wxdm(ctx: ParseContext) -> None:
    """Extract acquisition parameters from the ``WXDM`` block.

    Sets ``ctx.params["ExposureTime"]`` (float, seconds) when the key is
    present.  Missing or malformed blocks are silently ignored.
    """
    name = "WXDM"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        block = ctx.blocks[i]
        ctx.f.seek(block.offset + 16)
        payload = ctx.f.read(block.size - 16)

        props = decode_pset(payload)

        raw_ms = props.get("Exposure Time")
        if raw_ms is not None:
            try:
                ctx.params["ExposureTime"] = int(raw_ms) / 1000.0
            except (TypeError, ValueError):
                pass
