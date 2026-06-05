# -*- coding: utf-8 -*-
"""First-pass scan: block names, sizes, and file offsets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binary_io import BLOCK_HEADER_DTYPE


@dataclass(frozen=True)
class BlockInfo:
    """Metadata for a single WDF block (tag, byte size, file offset)."""

    name: str
    size: int
    offset: int


def scan_blocks(file_obj, filesize: int) -> list[BlockInfo]:
    """Walk the file and collect each block's tag, size, and starting
    offset."""
    blocks: list[BlockInfo] = []
    offset = 0
    while offset < filesize - 1:
        file_obj.seek(offset)
        block_header = np.fromfile(file_obj, dtype=BLOCK_HEADER_DTYPE, count=1)
        block_size = block_header["block_size"][0]
        if block_size == 0:
            break
        blocks.append(
            BlockInfo(
                name=block_header["block_name"][0].decode(),
                size=block_size,
                offset=offset,
            )
        )
        offset += block_size
    return blocks


def indices_named(blocks: list[BlockInfo], name: str) -> list[int]:
    """Return indices of blocks whose four-letter tag equals ``name``."""
    return [i for i, b in enumerate(blocks) if b.name == name]
