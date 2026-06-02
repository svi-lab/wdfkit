# -*- coding: utf-8 -*-
"""Parse ``WHTL`` embedded white-light image block."""

from __future__ import annotations

import io

from PIL import Image, ImageFile

from .._helpers.block_index import indices_named
from .._helpers.parse_context import ParseContext

_JPEG_SOI = b"\xff\xd8\xff"
_JPEG_EOI = b"\xff\xd9"


def parse_whtl(ctx: ParseContext) -> None:
    name = "WHTL"
    for i in indices_named(ctx.blocks, name):
        ctx.print_block_header(name, i)
        body_size = ctx.blocks["BlockSizes"][i] - 16
        ctx.f.seek(ctx.blocks["BlockOffsets"][i] + 16)
        raw_bytes = ctx.f.read(body_size)

        # Extract the raw JPEG payload (FF D8 FF ... FF D9).
        soi = raw_bytes.find(_JPEG_SOI)
        eoi = raw_bytes.rfind(_JPEG_EOI)
        if soi >= 0 and eoi > soi:
            ctx.whtl_jpeg_bytes = raw_bytes[soi : eoi + 2]
        else:
            ctx.whtl_jpeg_bytes = raw_bytes

        # Allow truncated JPEG/PNG thumbnails embedded in some WDF files.
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        ctx.img = Image.open(io.BytesIO(raw_bytes))
